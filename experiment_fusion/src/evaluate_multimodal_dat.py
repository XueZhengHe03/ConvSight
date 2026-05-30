# src/evaluate_multimodal_dat.py
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score
from model_dat import MultimodalFusionModel
from dataset_dat import MultimodalDataset

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备：{device}")

    # 测试集
    test_dataset = MultimodalDataset("../dat_new", split='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    print(f"📊 测试集：{len(test_dataset)}")

    # 加载模型
    model = MultimodalFusionModel(seq_len=2000, d_model=32, num_classes=2, enc_in=1)
    model_path = "../dataset_new/best_multimodal_dat.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("❌ 模型文件不存在")
        return
    model.to(device)
    model.eval()

    # 评估
    test_preds, test_labels = [], []
    with torch.no_grad():
        for img, ts, label in test_loader:
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            logits = model(img, ts)
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(label.cpu().tolist())

    # 计算指标
    test_acc = accuracy_score(test_labels, test_preds)
    recall_conv = recall_score(test_labels, test_preds, pos_label=1)
    recall_unconv = recall_score(test_labels, test_preds, pos_label=0)

    print(f"\n✅ 多模态融合模型测试结果 (dat dataset):")
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Converged Recall: {recall_conv:.4f}")
    print(f"  Unconverged Recall: {recall_unconv:.4f}")

if __name__ == "__main__":
    evaluate()
