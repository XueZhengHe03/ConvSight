# src/prepare_dataset_tslib_dat.py
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dat(path, use_last_n=2000):
    """加载 .dat 文件，提取 residual 列（单通道）"""
    # 跳过前4行标题，空格分隔
    data = np.loadtxt(path, skiprows=4)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    seq = data[:, 1:2].astype(np.float32)  # 只取 residual 列
    if len(seq) < use_last_n:
        pad = np.tile(seq[0], (use_last_n - len(seq), 1))
        seq = np.vstack([pad, seq])
    else:
        seq = seq[-use_last_n:]
    return seq

def normalize_per_series(seq):
    """每条序列独立标准化"""
    scaler = StandardScaler()
    return scaler.fit_transform(seq).astype(np.float32)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "dat_new")
    dataset_dir = os.path.join(project_root, "dataset_new", "tslib")
    os.makedirs(dataset_dir, exist_ok=True)

    # 加载数据
    X, y = [], []
    folders = [
        (os.path.join(data_root, "dat_converge"), 1),
        (os.path.join(data_root, "dat_un-converge"), 0)
    ]
    for folder, label in folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        print(f"Loading from {folder} (label={label})...")
        count = 0
        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.dat'):
                path = os.path.join(folder, fname)
                try:
                    seq = load_dat(path, use_last_n=2000)
                    seq = normalize_per_series(seq)
                    X.append(seq)
                    y.append(label)
                    count += 1
                except Exception as e:
                    print(f"  Skip {fname}: {e}")
        print(f"  Loaded {count} files.")

    X = np.stack(X)
    y = np.array(y)
    print(f"Loaded {len(X)} samples. Shape: {X.shape}")

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # 保存
    np.save(os.path.join(dataset_dir, "X_train.npy"), X_train)
    np.save(os.path.join(dataset_dir, "y_train.npy"), y_train)
    np.save(os.path.join(dataset_dir, "X_val.npy"), X_val)
    np.save(os.path.join(dataset_dir, "y_val.npy"), y_val)
    np.save(os.path.join(dataset_dir, "X_test.npy"), X_test)
    np.save(os.path.join(dataset_dir, "y_test.npy"), y_test)

    print(f"\n✅ Dataset saved to: {dataset_dir}")

if __name__ == "__main__":
    main()
