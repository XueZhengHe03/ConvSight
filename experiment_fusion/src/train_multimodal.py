# src/train_multimodal.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import MultimodalFusionModel
# from model_new import MultimodalFusionModel
from dataset import MultimodalDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备：{device}")

    # 超参数调整：刻意限制收敛速度
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-5       # [修改 1] 学习率降低，防止 10 epoch 内完全收敛
    weight_decay = 1e-2        # [修改 2] 权重衰减增大，限制权重幅度

    # 数据集
    train_dataset = MultimodalDataset("../dat", split='train')
    val_dataset = MultimodalDataset("../dat", split='val')
    test_dataset = MultimodalDataset("../dat", split='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"📊 训练集：{len(train_dataset)} | 验证集：{len(val_dataset)} | 测试集：{len(test_dataset)}")

    # 模型 (注意 d_model 需与 model.py 一致)
    model = MultimodalFusionModel(seq_len=2000, d_model=32, num_classes=2)
    model.to(device)

    # 优化器
    # [修改 3] 增加标签平滑，防止模型对训练数据过于自信
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # [修改 4] 学习率调度更温和，避免后期剧烈下降导致欠拟合过早
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_preds, train_labels = [], []
        for img, ts, label in train_loader:
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(img, ts)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(label.cpu().tolist())

        train_acc = accuracy_score(train_labels, train_preds)

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for img, ts, label in val_loader:
                img, ts, label = img.to(device), ts.to(device), label.to(device)
                logits = model(img, ts)
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(label.cpu().tolist())
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step()

        print(f"Epoch {epoch+1}/10 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 确保路径存在
            os.makedirs("../dataset", exist_ok=True)
            torch.save(model.state_dict(), "../dataset/best_multimodal.pth")
            # torch.save(model.state_dict(), "../dataset/best_multimodal_new.pth")

    # 最终测试
    if os.path.exists("../dataset/best_multimodal.pth"):
        model.load_state_dict(torch.load("../dataset/best_multimodal.pth", map_location=device))
        # model.load_state_dict(torch.load("../dataset/best_multimodal_new.pth", map_location=device))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for img, ts, label in test_loader:
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            logits = model(img, ts)
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(label.cpu().tolist())
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"\n🎯 最终测试准确率：{test_acc:.4f}")

if __name__ == "__main__":
    train()