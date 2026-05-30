# src/train_multimodal_new_dat.py
# 使用 Dynamic Gated Fusion 模型在 DAT 数据集上训练
# 采用与CSV数据集相同的训练思路
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score
from model_new import MultimodalFusionModel
from dataset_dat import MultimodalDataset

def train():
    device = torch.device("cpu")
    print(f"🚀 使用设备：{device}")

    # 超参数 (与CSV实验保持一致)
    batch_size = 16
    num_epochs = 10  # 与CSV实验保持一致
    learning_rate = 1e-5
    weight_decay = 1e-2

    # 数据集
    train_dataset = MultimodalDataset("../dat_new", split='train')
    val_dataset = MultimodalDataset("../dat_new", split='val')
    test_dataset = MultimodalDataset("../dat_new", split='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"📊 训练集：{len(train_dataset)} | 验证集：{len(val_dataset)} | 测试集：{len(test_dataset)}")

    # 统计类别分布
    train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    print(f"   训练集分布: 收敛={sum(train_labels)}, 未收敛={len(train_labels)-sum(train_labels)}")

    # 模型 (Dynamic Gated Fusion, d_model=64, enc_in=2)
    # 将DAT单通道数据复制为2通道以匹配CSV训练的模型结构
    model = MultimodalFusionModel(seq_len=2000, d_model=64, num_classes=2)
    model.to(device)

    # 优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_epoch = 0

    print("\n开始训练 Dynamic Gated Fusion (DAT数据集)...")
    print("=" * 60)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_preds, train_labels = [], []
        train_loss = 0.0
        for img, ts, label in train_loader:
            # 将单通道复制为2通道
            ts = ts.repeat(1, 1, 2)  # (B, T, 1) -> (B, T, 2)
            img, ts, label = img.to(device), ts.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img, ts)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(label.cpu().tolist())
            train_loss += loss.item()

        train_acc = accuracy_score(train_labels, train_preds)

        # 验证阶段
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for img, ts, label in val_loader:
                ts = ts.repeat(1, 1, 2)
                img, ts, label = img.to(device), ts.to(device), label.to(device)
                logits = model(img, ts)
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(label.cpu().tolist())
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step()

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            os.makedirs("../dataset_new", exist_ok=True)
            torch.save(model.state_dict(), "../dataset_new/best_multimodal_new_dat.pth")

    print("=" * 60)
    print(f"✅ 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    # 最终测试
    print("\n" + "=" * 60)
    print("最终测试结果:")
    print("=" * 60)

    if os.path.exists("../dataset_new/best_multimodal_new_dat.pth"):
        model.load_state_dict(torch.load("../dataset_new/best_multimodal_new_dat.pth", map_location=device))
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for img, ts, label in test_loader:
            ts = ts.repeat(1, 1, 2)
            img, ts, label = img.to(device), ts.to(device), label.to(device)
            logits = model(img, ts)
            preds = logits.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(label.cpu().tolist())

    test_acc = accuracy_score(test_labels, test_preds)
    recall_conv = recall_score(test_labels, test_preds, pos_label=1)
    recall_unconv = recall_score(test_labels, test_preds, pos_label=0)

    print(f"总体准确率:     {test_acc:.4f}")
    print(f"收敛样本召回率: {recall_conv:.4f}")
    print(f"未收敛样本召回率: {recall_unconv:.4f}")

    return test_acc, recall_conv, recall_unconv

if __name__ == "__main__":
    train()
