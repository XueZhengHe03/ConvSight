# src/data_utils_dat_mean.py
# Mean Centering 版本的数据加载工具
import os
import numpy as np

def load_timeseries(dat_path, use_last_n=2000):
    """加载 .dat 文件，提取 residual 列（单通道）"""
    data = np.loadtxt(dat_path, skiprows=4)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ts = data[:, 1:2].astype(np.float32)
    if len(ts) < use_last_n:
        pad = np.tile(ts[0], (use_last_n - len(ts), 1))
        ts = np.vstack([pad, ts])
    else:
        ts = ts[-use_last_n:]
    return ts

def normalize_per_series(ts):
    """仅做中心化（减去均值），保留振荡幅度信息"""
    mean = np.mean(ts, axis=0, keepdims=True)
    return (ts - mean).astype(np.float32)
