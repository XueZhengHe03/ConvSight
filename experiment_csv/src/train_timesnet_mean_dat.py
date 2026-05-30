# src/train_timesnet_mean_dat.py
# 使用 Mean Centering 预处理的数据训练 TimesNet
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "Time-Series-Library"))
from models.TimesNet import Model as TimesNet

class Configs:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 2000
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = 1
        self.num_class = 2
        self.d_model = 128
        self.d_ff = 128
        self.top_k = 5
        self.num_kernels = 6
        self.e_layers = 2
        self.dropout = 0.1
        self.embed = 'timeF'
        self.freq = 'h'
        self.c_out = 2

def main():
    dataset_dir = os.path.join(project_root, "dataset_new", "tslib_mean")
    X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
    y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    configs = Configs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TimesNet(configs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "best_timesnet_model_mean_dat.pt")
    best_val_acc = 0

    print("\nStart training TimesNet (Mean Centering, dat dataset)...")
    for epoch in tqdm(range(50), desc="Epochs"):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_mark_enc = torch.ones(x_batch.shape[0], x_batch.shape[1]).to(device)
            optimizer.zero_grad()
            logits = model(x_batch, x_mark_enc, None, None)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                x_mark_enc = torch.ones(x_batch.shape[0], x_batch.shape[1]).to(device)
                pred = model(x_batch, x_mark_enc, None, None).argmax(dim=1)
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
