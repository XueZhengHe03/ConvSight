# src/test_model.py
import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from dataset import MultimodalDataset
from model import MultimodalFusionModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 加载测试集
    test_dataset = MultimodalDataset("../dat", split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    print(f"📊 测试集样本数: {len(test_dataset)}")

    # 加载模型
    model = MultimodalFusionModel()
    ckpt_path = "../dataset/best_multimodal.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    print("✅ 模型加载成功")

    # 推理
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img, ts, label in test_loader:
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            logits = model(img, ts)
            pred = logits.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # 计算指标
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall_converged = tp / (tp + fn) if (tp + fn) > 0 else 0.0      # 收敛召回率
    recall_unconverged = tn / (tn + fp) if (tn + fp) > 0 else 0.0    # 未收敛召回率

    print("\n" + "="*50)
    print("🎯 测试集性能评估")
    print("="*50)
    print(f"总体准确率 (Accuracy):       {acc:.4f} ({acc*100:.2f}%)")
    print(f"收敛数据召回率 (Recall↑):   {recall_converged:.4f} ({recall_converged*100:.2f}%)")
    print(f"未收敛数据召回率 (Recall↓): {recall_unconverged:.4f} ({recall_unconverged*100:.2f}%)")
    print("\n📋 分类报告:")
    print(classification_report(labels, preds, target_names=["Unconverged", "Converged"]))
    print("🧮 混淆矩阵 (Actual \\ Predicted):")
    print(cm)


if __name__ == "__main__":
    main()