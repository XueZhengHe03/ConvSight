# src/infer_new_csv_with_teacher.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
from scipy import stats

def check_hard_rules(v1, v2, v3):
    """
    [核心修复] 基于文档 §6.3 的硬规则预检查
    目的：防止模型将“长序列缓慢发散”误判为收敛。
    规则：如果曲线末端呈现明显的单调上扬或下降，直接判定为未收敛。
    """
    def is_monotonic_trend(v, threshold_slope=1e-4, monotonic_ratio=0.8):
        if len(v) < 100:
            return False
        
        # 取最后 2000 步或全部
        tail_v = v[-2000:] if len(v) > 2000 else v
        tail_steps = len(tail_v)
        
        # 1. 线性回归检测斜率
        x = np.arange(tail_steps)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, tail_v)
        
        # 如果斜率绝对值过大，认为有单调趋势
        # 注意：由于数据经过脱密缩放，这里使用相对斜率 (斜率 / 数据范围)
        data_range = tail_v.max() - tail_v.min()
        if data_range == 0: data_range = 1.0
        relative_slope = abs(slope * tail_steps) / data_range
        
        if relative_slope > 0.15: # 阈值：如果在最后2000步内变化超过总波动范围的15%，视为发散
            return True
            
        # 2. 单调性比例检测 (防止非线性但单向的趋势)
        diffs = np.diff(tail_v)
        positive_ratio = np.sum(diffs > 0) / len(diffs)
        negative_ratio = np.sum(diffs < 0) / len(diffs)
        
        if positive_ratio > monotonic_ratio or negative_ratio > monotonic_ratio:
            return True
            
        return False

    # 只要 v2 或 v3 任意一个呈现明显单调趋势，即为未收敛
    if is_monotonic_trend(v2) or is_monotonic_trend(v3):
        return False # 未收敛
    
    return True # 通过硬规则，交给模型判断

def plot_curve(v1, v2, mode='global'):
    """根据文档 §6.a 生成单张曲线图"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if len(v1) == 0 or len(v2) == 0:
        ax.axis('off')
        return fig_to_pil(fig)

    if mode == 'global':
        ax.plot(v1, v2, linewidth=1.0)
        ax.set_xlim(v1.min(), v1.max())
        ax.set_ylim(v2.min(), v2.max())
    elif mode == 'local':
        N = float(v1[-1])
        if N <= 0: N = len(v1)
        
        # 1. 取最后 2000 步平均值 Vavr
        threshold = max(N - 2000, v1.min())
        last_2000_mask = v1 >= threshold
        if not np.any(last_2000_mask):
            last_2000_mask = np.ones_like(v1, dtype=bool)
        Vavr = np.mean(v2[last_2000_mask])
        
        # 2. 在 (1000, N) 区间找极值
        start_threshold = max(1000.0, v1.min())
        segment_mask = (v1 >= start_threshold) & (v1 <= N)
        if not np.any(segment_mask):
            segment_mask = np.ones_like(v1, dtype=bool)
            
        segment = v2[segment_mask]
        if len(segment) == 0: segment = v2
        
        Vmax, Vmin = segment.max(), segment.min()
        
        # 文档公式：range = MAX(Vmax - Vmin, 0.005, |Vavr| * 0.01)
        range_val = max(Vmax - Vmin, 0.005, abs(Vavr) * 0.01)
        
        # [优化] 防止 range 过小导致视觉欺骗，增加一个基于整体波动的下限
        global_range = v2.max() - v2.min()
        if global_range > 0:
            range_val = max(range_val, global_range * 0.05) # 至少显示全局波动的5%
        
        ax.plot(v1, v2, linewidth=1.0)
        ax.set_xlim(1, N)
        ax.set_ylim(Vavr - range_val, Vavr + range_val)

    ax.axis('off')
    fig.tight_layout(pad=0)
    return fig_to_pil(fig)

def fig_to_pil(fig):
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', (width, height), buf, 'raw', 'RGBA', 0, 1)
    img = img.convert('RGB')
    plt.close(fig)
    return img

def generate_four_in_one_image_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # 兼容列名
    if 'v1' in df.columns:
        v1 = df['v1'].values
        v2 = df['v2'].values
        v3 = df['v3'].values
    else:
        v1 = df.iloc[:, 0].values
        v2 = df.iloc[:, 1].values
        v3 = df.iloc[:, 2].values

    subplots = []
    for var in [v2, v3]:
        for mode in ['global', 'local']:
            img = plot_curve(v1, var, mode=mode)
            subplots.append(img)

    w, h = subplots[0].size
    combined = Image.new('RGB', (w * 2, h * 2))
    combined.paste(subplots[0], (0, 0))
    combined.paste(subplots[1], (w, 0))
    combined.paste(subplots[2], (0, h))
    combined.paste(subplots[3], (w, h))
    return combined

def preprocess_timeseries(csv_path, seq_len=2000):
    """
    加载并预处理时序数据
    [修改] 不再做 Z-Score 标准化，避免消除单调趋势特征。
    仅做长度对齐和简单的幅值缩放，保留原始形状和相对趋势。
    """
    df = pd.read_csv(csv_path)
    if 'v2' in df.columns:
        v2 = df['v2'].values.astype(np.float32)
        v3 = df['v3'].values.astype(np.float32)
    else:
        v2 = df.iloc[:, 1].values.astype(np.float32)
        v3 = df.iloc[:, 2].values.astype(np.float32)
        
    ts = np.stack([v2, v3], axis=1)

    # 1. 长度对齐：取最后 seq_len 步 (文档强调关注末端)
    if len(ts) < seq_len:
        pad_len = seq_len - len(ts)
        first_val = ts[0]
        pad = np.tile(first_val, (pad_len, 1))
        ts = np.vstack([pad, ts])
    else:
        ts = ts[-seq_len:]

    # 2. 幅值缩放 (Scale to [-1, 1]) 但不平移均值，或者只做简单的归一化
    # 这里采用 Min-Max 到 [-1, 1]，这不会改变曲线的单调性（斜率符号不变）
    min_vals = ts.min(axis=0, keepdims=True)
    max_vals = ts.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    ts_norm = 2.0 * (ts - min_vals) / range_vals - 1.0
    return ts_norm.astype(np.float32)

def main():
    if len(sys.argv) != 2:
        print("用法：python infer_new_csv_with_teacher.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在：{csv_path}")
        sys.exit(1)

    print(f"🚀 开始分析：{os.path.basename(csv_path)}")

    # 读取数据用于硬规则检查
    df = pd.read_csv(csv_path)
    if 'v1' in df.columns:
        v1, v2, v3 = df['v1'].values, df['v2'].values, df['v3'].values
    else:
        v1, v2, v3 = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values

    # --- [核心步骤 1] 硬规则预检查 ---
    print("⏳ 正在执行物理规则预检查 (文档 §6.3)...")
    if not check_hard_rules(v1, v2, v3):
        print("\n" + "="*40)
        print("🛑 触发硬规则拦截")
        print("="*40)
        print("  检测结果：曲线末端存在明显的单调上扬或下降趋势")
        print("  判定结果：❌ 未收敛 (Unconverged)")
        print("  原因：违反文档 §6.3 '没有呈现明显的单边上扬或者下降' 原则")
        print("="*40)
        return # 直接结束，不调用模型

    print("✅ 通过硬规则检查，进入模型推理...")

    # 1. 生成四合一图像
    try:
        img = generate_four_in_one_image_from_csv(csv_path)
        print("✅ 四合一图像生成成功")
    except Exception as e:
        print(f"❌ 图像生成失败：{e}")
        return

    # 2. 预处理时序
    try:
        ts_data = preprocess_timeseries(csv_path, seq_len=2000)
        print("✅ 时序数据预处理完成")
    except Exception as e:
        print(f"❌ 时序处理失败：{e}")
        return

    # 3. 转换为 Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).numpy()
    ts_tensor = np.expand_dims(ts_data, axis=0)

    # 4. ONNX 推理
    onnx_path = "../dataset/teacher_multimodal_new.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ 未找到 ONNX 模型：{onnx_path}")
        return

    try:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        outputs = sess.run(None, {"image": img_tensor, "timeseries": ts_tensor})
        logits = outputs[0][0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        prob_converge = probs[1]
        is_converged = prob_converge > 0.5

        print("\n" + "="*40)
        print("🎯 教师模型判断结果")
        print("="*40)
        print(f"  收敛概率：{prob_converge:.4f} ({prob_converge*100:.2f}%)")
        print(f"  最终结论：{'✅ 收敛 (Converged)' if is_converged else '❌ 未收敛 (Unconverged)'}")
        print("="*40)

    except Exception as e:
        print(f"❌ ONNX 推理失败：{e}")

if __name__ == "__main__":
    main()