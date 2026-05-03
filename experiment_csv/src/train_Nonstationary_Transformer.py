# src/train_Nonstationary_Transformer.py
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "Time-Series-Library"))
from models.Nonstationary_Transformer import Model as NonstationaryTransformer

class Configs:
    def __init__(self):
        # 任务相关
        self.task_name = 'classification'
        self.seq_len = 2000
        self.label_len = 0      # 必须存在
        self.pred_len = 0
        self.enc_in = 2         # v2, v3
        self.num_class = 2
        # 模型结构
        self.d_model = 128
        self.d_ff = 128
        self.e_layers = 2
        self.dropout = 0.1
        self.embed = 'timeF'
        self.freq = 'h'
        self.c_out = 2
        self.factor = 3
        self.n_heads = 8
        self.activation = 'gelu'
        # Projector 参数（必须存在）
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2

def main():
    # 加载数据
    dataset_dir = os.path.join(project_root, "dataset", "tslib")
    X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
    y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))

    # 转为 Tensor
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    # 创建 x_mark_enc（全1，表示所有时间步有效）
    x_mark_enc_train = torch.ones(X_train.shape[0], X_train.shape[1])
    x_mark_enc_val = torch.ones(X_val.shape[0], X_val.shape[1])

    # DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train_t, x_mark_enc_train, y_train_t)
    val_dataset = TensorDataset(X_val_t, x_mark_enc_val, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型
    configs = Configs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NonstationaryTransformer(configs).to(device)

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "best_Nonstationary_Transformer_model.pt")
    best_val_acc = 0

    print("\nStart training Non-stationary Transformer...")
    for epoch in tqdm(range(50), desc="Epochs"):
        model.train()
        total_loss = 0
        for x_batch, x_mark_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            x_mark_batch = x_mark_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch, x_mark_batch, None, None)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, x_mark_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                x_mark_batch = x_mark_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch, x_mark_batch, None, None).argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0 or epoch == 49:
            tqdm.write(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    print(f"\n✅ Model saved to: {model_path}")

if __name__ == "__main__":
    main()