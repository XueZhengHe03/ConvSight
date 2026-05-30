# src/train_ts2vec_dat.py
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "ts2vec"))
from ts2vec import TS2Vec

def main():
    # 加载数据
    dataset_dir = os.path.join(project_root, "dataset_new", "tslib")
    X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
    y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 预训练
    print("Initializing TS2Vec...")
    ts2vec_model = TS2Vec(input_dims=1, output_dims=320, hidden_dims=64, depth=10, device='cpu')  # 修改：input_dims=1
    print("Pre-training TS2Vec...")
    ts2vec_model.fit(X_train, verbose=True, n_epochs=10)

    # 转 Tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    # 分类器
    class ConvergenceClassifier(nn.Module):
        def __init__(self, backbone, example_input):
            super().__init__()
            self.backbone = backbone
            rep = self.backbone.encode(example_input.cpu().numpy(), encoding_window='full_series')
            D = rep.shape[1]
            self.classifier = nn.Linear(D, 2)
            print(f"✅ Embedding dim: {D}")

        def forward(self, x):
            x_np = x.cpu().numpy()
            rep = self.backbone.encode(x_np, encoding_window='full_series')
            rep = torch.from_numpy(rep).to(x.device)
            return self.classifier(rep)

    model = ConvergenceClassifier(ts2vec_model, X_train_t[:1]).to(device)

    # 训练
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model_path = os.path.join(checkpoint_dir, "best_convergence_model_dat.pt")
    best_val_acc = 0

    print("\nStart training TS2Vec (dat dataset)...")
    for epoch in tqdm(range(50), desc="Epochs"):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val_t).argmax(dim=1) == y_val_t).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0 or epoch == 49:
            tqdm.write(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    print(f"\n✅ Model saved to: {model_path}")

if __name__ == "__main__":
    main()
