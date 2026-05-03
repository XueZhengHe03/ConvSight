# src/utils.py
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_timeseries(csv_path, use_last_n=2000):
    """加载 v2, v3，取最后 use_last_n 步"""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)  # 跳过 header
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ts = data[:, 1:3].astype(np.float32)  # v2, v3
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