# src/cross_validation_teacher.py
import sys
import os
import gc
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm  # 用于显示进度条

# [修改 1] 从 model_new 导入最新的动态门控融合模型
from model_new import MultimodalFusionModel

# 确保数据集脚本存在
try:
    from cv_dataset import CVMultimodalDataset
    from prepare_cv_data import get_all_samples
except ImportError as e:
    print(f"❌ 导入数据集模块失败：{e}")
    print("   请确认 src/cv_dataset.py 和 src/prepare_cv_data.py 是否存在且正确。")
    sys.exit(1)

def train_and_evaluate_on_fold(train_samples, train_labels, val_samples, val_labels, fold, device):
    print(f"\n🔄 开始训练 Fold {fold+1}...")
    
    # 创建数据集
    train_dataset = CVMultimodalDataset(train_samples, train_labels)
    val_dataset = CVMultimodalDataset(val_samples, val_labels)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # [修改 2] 初始化模型：关键参数必须与 train_multimodal.py 一致
    # d_model=32, seq_len=2000
    model = MultimodalFusionModel(seq_len=2000, d_model=32, num_classes=2)
    model.to(device)
    
    # [修改 3] 优化器与损失函数升级
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) # 标签平滑防止过拟合
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    num_epochs = 10
    best_val_acc = 0.0
    best_model_state = None

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}", leave=False)
        
        for img, ts, label in pbar:
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = model(img, ts)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # 每个 epoch 进行简单验证，保存最佳模型
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img, ts, label in val_loader:
                img, ts, label = img.to(device), ts.to(device), label.to(device)
                logits = model(img, ts)
                preds = logits.argmax(dim=1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
        
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            
        if (epoch + 1) % 5 == 0:
            print(f"  [{fold+1}] Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")

    # 加载最佳模型进行最终评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 详细评估
    model.eval()
    all_preds, all_labels_list = [], []
    
    with torch.no_grad():
        for img, ts, label in val_loader:
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            logits = model(img, ts)
            pred = logits.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels_list.append(label.cpu())
    
    preds = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels_list).numpy()
    
    # 保存该 fold 的模型
    os.makedirs("../cv_models", exist_ok=True)
    torch.save(model.state_dict(), f"../cv_models/fold_{fold+1}_teacher_new.pth")
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_val_acc, preds, labels_np

def main():
    print("🚀 开始 5 折交叉验证评估教师模型 (Model_New: Dynamic Gated Fusion)...")
    
    # 设备检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 使用设备：{device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 准备数据
    try:
        samples, labels = get_all_samples()
    except Exception as e:
        print(f"❌ 获取样本失败：{e}")
        return

    print(f"📊 样本总数：{len(samples)}")
    if len(samples) < 5:
        print("❌ 样本数量不足，无法进行 5 折交叉验证。")
        return
    
    # 转为 NumPy 数组
    samples = np.array(samples)
    labels = np.array(labels)
    
    # 5 折分层划分 (保证每折中正负样本比例一致)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    all_preds, all_labels_global = [], []

    # 存储每折的详细指标
    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
        train_samples = samples[train_idx].tolist()
        train_labels = labels[train_idx].tolist()
        val_samples = samples[val_idx].tolist()
        val_labels = labels[val_idx].tolist()

        acc, preds, labels_fold = train_and_evaluate_on_fold(
            train_samples, train_labels, val_samples, val_labels, fold, device
        )
        
        cv_scores.append(acc)
        all_preds.extend(preds)
        all_labels_global.extend(labels_fold)
        
        # 计算每折的召回率
        tn, fp, fn, tp = confusion_matrix(labels_fold, preds, labels=[0, 1]).ravel()
        rec_conv = tp / (tp + fn) if (tp + fn) > 0 else 0
        rec_unconv = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"✅ Fold {fold+1} 完成 -> Acc: {acc:.4f} | Recall(Conv): {rec_conv:.4f} | Recall(Unconv): {rec_unconv:.4f}")

    # 最终结果汇总
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    overall_acc = (np.array(all_preds) == np.array(all_labels_global)).mean()
    
    # 全局混淆矩阵
    cm = confusion_matrix(all_labels_global, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    global_rec_conv = tp / (tp + fn) if (tp + fn) > 0 else 0
    global_rec_unconv = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "="*60)
    print("🎯 5 折交叉验证最终结果 (Model_New)")
    print("="*60)
    print(f"各折准确率：{[f'{x:.4f}' for x in cv_scores]}")
    print(f"平均准确率：{mean_acc:.4f} ± {std_acc:.4f}")
    print(f"整体准确率：{overall_acc:.4f}")
    print("-" * 60)
    print(f"全局收敛召回率：{global_rec_conv:.4f}")
    print(f"全局未收敛召回率：{global_rec_unconv:.4f}")
    print("-" * 60)
    print("🧮 全局混淆矩阵:")
    print(f"             Pred-Unconv  Pred-Conv")
    print(f"Act-Unconv     {tn:5d}       {fp:5d}")
    print(f"Act-Conv       {fn:5d}       {tp:5d}")
    print("="*60)
    
    # 保存结果到文件
    result_file = "../cv_results_summary.txt"
    with open(result_file, "w") as f:
        f.write(f"Model: MultimodalFusionModel (Dynamic Gated Fusion, d_model=32)\n")
        f.write(f"Total Samples: {len(samples)}\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write(f"Recall (Converged): {global_rec_conv:.4f}\n")
        f.write(f"Recall (Unconverged): {global_rec_unconv:.4f}\n")
        f.write(f"\nFold Scores: {cv_scores}\n")
    print(f"📄 详细结果已保存至：{result_file}")

if __name__ == "__main__":
    main()