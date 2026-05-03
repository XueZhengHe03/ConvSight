# src/evaluate_timesnet.py
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "Time-Series-Library"))
from models.TimesNet import Model as TimesNet

class Configs:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 2000
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = 2
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
    # 路径
    dataset_dir = os.path.join(project_root, "dataset", "tslib")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoint_dir, "best_timesnet_model.pt")

    # 加载测试集
    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    # 模型
    configs = Configs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TimesNet(configs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # DataLoader
    batch_size = 32
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    # 评估
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_mark_enc = torch.ones(x_batch.shape[0], x_batch.shape[1]).to(device)
            pred = model(x_batch, x_mark_enc, None, None).argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_acc = (all_preds == all_labels).float().mean().item()
    acc_conv = (all_preds[all_labels == 1] == 1).float().mean().item()
    acc_unconv = (all_preds[all_labels == 0] == 0).float().mean().item()

    print(f"\n✅ TimesNet Final Test Results:")
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Converged Recall:   {acc_conv:.4f}")
    print(f"  Unconverged Recall: {acc_unconv:.4f}")

if __name__ == "__main__":
    main()