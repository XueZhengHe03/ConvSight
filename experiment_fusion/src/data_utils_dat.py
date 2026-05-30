# src/data_utils_dat.py
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_timeseries(dat_path, use_last_n=2000):
    """加载 .dat 文件，提取 residual 列（单通道）"""
    data = np.loadtxt(dat_path, skiprows=4)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ts = data[:, 1:2].astype(np.float32)  # 只取 residual 列
    if len(ts) < use_last_n:
        pad = np.tile(ts[0], (use_last_n - len(ts), 1))
        ts = np.vstack([pad, ts])
    else:
        ts = ts[-use_last_n:]
    return ts

def normalize_per_series(ts):
    """每条序列独立标准化"""
    scaler = StandardScaler()
    return scaler.fit_transform(ts).astype(np.float32)
