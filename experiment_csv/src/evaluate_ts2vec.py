# src/evaluate_ts2vec.py
import sys
import os
import numpy as np
import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "ts2vec"))
from ts2vec import TS2Vec

class ConvergenceClassifier(nn.Module):
    def __init__(self, backbone, D):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(D, 2)

    def forward(self, x):
        x_np = x.cpu().numpy()
        rep = self.backbone.encode(x_np, encoding_window='full_series')
        rep = torch.from_numpy(rep).to(x.device)
        return self.classifier(rep)

def main():
    # 模型路径
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    model_path = os.path.join(checkpoint_dir, "best_convergence_model.pt")
    
    # ✅ 正确路径：从 dataset/ts2vec/ 加载测试集
    dataset_dir = os.path.join(project_root, "dataset", "ts2vec")
    X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))
    print(f"Loaded test set from {dataset_dir}: {X_test.shape}")

    # 初始化 TS2Vec
    ts2vec_model = TS2Vec(
        input_dims=2,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cpu'
    )

    # 探测维度 D
    example_input = X_test[:1]
    rep = ts2vec_model.encode(example_input, encoding_window='full_series')
    D = rep.shape[1]
    print(f"Detected embedding dimension: {D}")

    # 构建模型并加载权重
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvergenceClassifier(ts2vec_model, D).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 转为 Tensor
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    # 测试
    with torch.no_grad():
        test_pred = model(X_test_tensor).argmax(dim=1)
        test_acc = (test_pred == y_test_tensor).float().mean().item()
        acc_conv = (test_pred[y_test_tensor == 1] == 1).float().mean().item()
        acc_unconv = (test_pred[y_test_tensor == 0] == 0).float().mean().item()

    print(f"\n✅ Final Test Results:")
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Converged Recall:   {acc_conv:.4f}")
    print(f"  Unconverged Recall: {acc_unconv:.4f}")

if __name__ == "__main__":
    main()