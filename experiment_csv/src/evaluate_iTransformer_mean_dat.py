# src/evaluate_iTransformer_mean_dat.py
# 评估 Mean Centering 预处理的 iTransformer
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "Time-Series-Library"))
from models.iTransformer import Model as iTransformer

class Configs:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 2000
        self.pred_len = 0
        self.enc_in = 1
        self.num_class = 2
        self.d_model = 128
        self.d_ff = 128
        self.e_layers = 3
        self.dropout = 0.1
        self.embed = 'timeF'
        self.freq = 'h'
        self.c_out = 2
        self.factor = 3
        self.n_heads = 8
        self.activation = 'gelu'

def main():
    # 路径
    dataset_dir = os.path.join(project_root, "dataset_new", "tslib_mean")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoint_dir, "best_iTransformer_model_mean_dat.pt")

    # 加载测试集
    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    # 模型
    configs = Configs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = iTransformer(configs).to(device)
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
            pred = model(x_batch, None, None, None).argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_acc = (all_preds == all_labels).float().mean().item()
    acc_conv = (all_preds[all_labels == 1] == 1).float().mean().item()
    acc_unconv = (all_preds[all_labels == 0] == 0).float().mean().item()

    print(f"\n✅ iTransformer Final Test Results (Mean Centering, dat dataset):")
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Converged Recall:   {acc_conv:.4f}")
    print(f"  Unconverged Recall: {acc_unconv:.4f}")

if __name__ == "__main__":
    main()
