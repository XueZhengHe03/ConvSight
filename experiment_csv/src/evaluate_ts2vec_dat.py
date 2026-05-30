# src/evaluate_ts2vec_dat.py
import sys
import os
import numpy as np
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "ts2vec"))
from ts2vec import TS2Vec

def main():
    # 路径
    dataset_dir = os.path.join(project_root, "dataset_new", "tslib")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoint_dir, "best_convergence_model_dat.pt")

    # 加载测试集
    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

    # 加载 TS2Vec backbone
    print("Loading TS2Vec backbone...")
    ts2vec_model = TS2Vec(input_dims=1, output_dims=320, hidden_dims=64, depth=10, device='cpu')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    # 分类器
    class ConvergenceClassifier(nn.Module):
        def __init__(self, backbone, example_input):
            super().__init__()
            self.backbone = backbone
            rep = self.backbone.encode(example_input.cpu().numpy(), encoding_window='full_series')
            D = rep.shape[1]
            self.classifier = nn.Linear(D, 2)

        def forward(self, x):
            x_np = x.cpu().numpy()
            rep = self.backbone.encode(x_np, encoding_window='full_series')
            rep = torch.from_numpy(rep).to(x.device)
            return self.classifier(rep)

    model = ConvergenceClassifier(ts2vec_model, X_test_t[:1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 评估
    with torch.no_grad():
        logits = model(X_test_t)
        all_preds = logits.argmax(dim=1)
        all_labels = y_test_t

    test_acc = (all_preds == all_labels).float().mean().item()
    acc_conv = (all_preds[all_labels == 1] == 1).float().mean().item()
    acc_unconv = (all_preds[all_labels == 0] == 0).float().mean().item()

    print(f"\n✅ TS2Vec Final Test Results (dat dataset):")
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Converged Recall:   {acc_conv:.4f}")
    print(f"  Unconverged Recall: {acc_unconv:.4f}")

if __name__ == "__main__":
    main()
