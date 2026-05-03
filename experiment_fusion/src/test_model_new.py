# src/test_model_new.py
import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from dataset import MultimodalDataset
# [修改 1] 从 model_new 导入改进后的模型
from model_new import MultimodalFusionModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备：{device}")

    # 加载测试集
    # 确保 ../dat 目录下有 test 分割的数据
    test_dataset = MultimodalDataset("../dat", split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    print(f"📊 测试集样本数：{len(test_dataset)}")

    # 加载模型
    # [修改 2] 显式指定参数，必须与训练时 (train_multimodal.py) 完全一致
    # 根据您的训练记录：d_model=32, seq_len=2000
    model = MultimodalFusionModel(seq_len=2000, d_model=32, num_classes=2)
    
    # [修改 3] 优先查找新模型的权重文件
    ckpt_path = "../dataset/best_multimodal_new.pth"
    if not os.path.exists(ckpt_path):
        # 如果新文件不存在，尝试旧文件名作为备选
        alt_path = "../dataset/best_multimodal.pth"
        if os.path.exists(alt_path):
            print(f"⚠️ 未找到 {ckpt_path}，尝试使用备用路径：{alt_path}")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"❌ 未找到模型权重文件。\n   请确认以下文件是否存在:\n   - {ckpt_path}\n   - {alt_path}")
            
    print(f"📂 正在加载权重：{ckpt_path}")
    
    # 加载状态字典 (处理可能的 key 不匹配或设备问题)
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print("   可能原因：model_new.py 的架构定义与训练保存时的架构不一致。")
        print("   请检查 d_model, seq_len 以及 DynamicGatedFusion 的维度定义。")
        return

    model.eval().to(device)
    print("✅ 模型加载成功")

    # 推理
    all_preds = []
    all_labels = []
    
    print("⏳ 正在进行测试集推理...")
    with torch.no_grad():
        for img, ts, label in test_loader:
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            
            # 前向传播
            logits = model(img, ts)
            pred = logits.argmax(dim=1)
            
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # 计算指标
    acc = (preds == labels).mean()
    
    # 混淆矩阵: [[TN, FP], [FN, TP]]
    # Label 0: Unconverged, Label 1: Converged
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 计算召回率
    # 收敛召回率 (Recall for Converged): 真正例 / (真正例 + 假负例)
    # 重要：避免漏判已收敛的计算，以节省算力
    recall_converged = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 未收敛召回率 (Recall for Unconverged): 真负例 / (真负例 + 假正例)
    # 重要：避免误判未收敛为收敛，导致结果错误
    recall_unconverged = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 输出结果
    print("\n" + "="*60)
    print("🎯 测试集性能评估 (Model: Dynamic Gated Fusion)")
    print("="*60)
    print(f"总体准确率 (Accuracy):       {acc:.4f} ({acc*100:.2f}%)")
    print(f"收敛数据召回率 (Recall↑):   {recall_converged:.4f} ({recall_converged*100:.2f}%)")
    print(f"未收敛数据召回率 (Recall↓): {recall_unconverged:.4f} ({recall_unconverged*100:.2f}%)")
    
    print("\n📋 分类报告:")
    # target_names 对应标签 0 和 1
    print(classification_report(labels, preds, target_names=["Unconverged", "Converged"], digits=4))
    
    print("🧮 混淆矩阵 (Actual \\ Predicted):")
    print("             Pred-Unconv  Pred-Conv")
    print(f"Act-Unconv     {tn:5d}       {fp:5d}")
    print(f"Act-Conv       {fn:5d}       {tp:5d}")
    
    # 简单分析
    if recall_converged < 0.90:
        print("\n⚠️ 警告：收敛召回率较低，可能会浪费大量计算资源（漏判收敛）。")
    if recall_unconverged < 0.90:
        print("\n⚠️ 警告：未收敛召回率较低，可能会产生错误的计算结果（误判收敛）。")

if __name__ == "__main__":
    main()