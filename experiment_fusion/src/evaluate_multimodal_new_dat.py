# src/evaluate_multimodal_new_dat.py
# 评估 Dynamic Gated Fusion 模型在 DAT 数据集上的表现
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, classification_report
from model_new import MultimodalFusionModel
from dataset_dat import MultimodalDataset

def evaluate():
    device = torch.device("cpu")

    # 测试集
    test_dataset = MultimodalDataset("../dat_new", split='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"📊 测试集：{len(test_dataset)}")

    # 加载模型
    model = MultimodalFusionModel(seq_len=2000, d_model=64, num_classes=2)
    model_path = "../dataset_new/best_multimodal_new_dat.pth"

    if not os.path.exists(model_path):
        print("❌ 模型文件不存在")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 评估
    test_preds, test_labels = [], []
    with torch.no_grad():
        for img, ts, label in test_loader:
            ts = ts.repeat(1, 1, 2)  # 单通道复制为2通道
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            logits = model(img, ts)
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(label.cpu().tolist())

    # 计算指标
    test_acc = accuracy_score(test_labels, test_preds)
    recall_conv = recall_score(test_labels, test_preds, pos_label=1)
    recall_unconv = recall_score(test_labels, test_preds, pos_label=0)

    print(f"\n✅ Dynamic Gated Fusion 测试结果 (DAT数据集):")
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Converged Recall: {recall_conv:.4f}")
    print(f"  Unconverged Recall: {recall_unconv:.4f}")

    print(f"\n详细分类报告:")
    print(classification_report(test_labels, test_preds, target_names=['Unconverged', 'Converged']))

if __name__ == "__main__":
    evaluate()
