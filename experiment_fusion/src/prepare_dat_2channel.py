# src/prepare_dat_2channel.py
# 将DAT数据转为2通道格式，用于跨域泛化实验
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dat_2channel(dat_path, use_last_n=2000):
    """加载 .dat 文件，将单通道复制为2通道"""
    data = np.loadtxt(dat_path, skiprows=4)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    seq = data[:, 1:2].astype(np.float32)  # 只取 residual 列
    # 复制为2通道
    seq_2ch = np.concatenate([seq, seq], axis=1)
    if len(seq_2ch) < use_last_n:
        pad = np.tile(seq_2ch[0], (use_last_n - len(seq_2ch), 1))
        seq_2ch = np.vstack([pad, seq_2ch])
    else:
        seq_2ch = seq_2ch[-use_last_n:]
    return seq_2ch

def normalize_per_series(seq):
    scaler = StandardScaler()
    return scaler.fit_transform(seq).astype(np.float32)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dat_root = os.path.join(project_root, "dat_new")
    output_dir = os.path.join(project_root, "dat_new_2ch")
    os.makedirs(output_dir, exist_ok=True)

    for category in ['dat_converge', 'dat_un-converge']:
        src_dir = os.path.join(dat_root, category)
        dst_dir = os.path.join(output_dir, category)
        os.makedirs(dst_dir, exist_ok=True)

        print(f"处理 {category}...")
        for fname in sorted(os.listdir(src_dir)):
            if fname.endswith('.dat'):
                src_path = os.path.join(src_dir, fname)
                seq = load_dat_2channel(src_path)
                seq = normalize_per_series(seq)
                # 保存为npy格式
                np.save(os.path.join(dst_dir, fname.replace('.dat', '.npy')), seq)

        print(f"  完成: {len(os.listdir(dst_dir))} 个文件")

    print(f"\n✅ 2通道数据已保存到: {output_dir}")

if __name__ == "__main__":
    main()
