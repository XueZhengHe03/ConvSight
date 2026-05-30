# src/dat2img_dat.py
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -----------------------------
# 配置
# -----------------------------
TASKS = [
    ("../dat_new/dat_converge", "../img_new/img_converge"),
    ("../dat_new/dat_un-converge", "../img_new/img_un-converge"),
]

# -----------------------------
# 局部视图参数
# -----------------------------
WINDOW_SIZE = 2000
MIN_X = 1000
RANGE_MIN_ABS = 0.005
RANGE_MIN_REL_FACTOR = 0.01

# -----------------------------
# 解析 .dat 文件
# -----------------------------
def load_dat_file(dat_path):
    """解析 .dat 文件，返回 (iter, residual)"""
    data = np.loadtxt(dat_path, skiprows=4)
    x = data[:, 0]
    residual = data[:, 1]
    return x, residual

# -----------------------------
# 处理单个 .dat 文件
# -----------------------------
def process_dat(dat_path: Path, img_dir: Path):
    try:
        x, residual = load_dat_file(dat_path)
        N = int(x[-1])

        def get_local_ylim(y):
            start_avg = max(N - WINDOW_SIZE, 1)
            idx_avg = np.searchsorted(x, start_avg, side='left')
            Vavr = np.mean(y[idx_avg:])

            if N <= MIN_X:
                start_range_val = max(N // 2, 1)
            else:
                start_range_val = MIN_X

            idx_range = np.searchsorted(x, start_range_val, side='left')
            segment = y[idx_range:]

            if len(segment) == 0:
                Vmax = Vmin = y[-1]
            else:
                Vmax = np.max(segment)
                Vmin = np.min(segment)

            range_val = max(
                Vmax - Vmin,
                RANGE_MIN_ABS,
                abs(Vavr) * RANGE_MIN_REL_FACTOR
            )
            return Vavr - range_val, Vavr + range_val, Vavr

        # 创建 1x2 子图（全局 + 局部）
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        stem = dat_path.stem
        fig.suptitle(f'Convergence Curves - {stem}', fontsize=16, fontweight='bold')

        # 全局视图
        axes[0].plot(x, residual, color='red', linewidth=1.0)
        axes[0].set_title('Global View')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Residual')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # 局部视图
        ymin, ymax, Vavr = get_local_ylim(residual)
        axes[1].plot(x, residual, color='purple', linewidth=1.0)
        axes[1].set_xlim(1, N)
        axes[1].set_ylim(ymin, ymax)
        axes[1].set_title(f'Local View\nVavr={Vavr:.5f}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Residual')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_path = img_dir / f"{stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 已生成: {output_path}")

    except Exception as e:
        print(f"❌ 处理 {dat_path.name} 时出错: {e}")

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    for dat_dir_str, img_dir_str in TASKS:
        dat_dir = Path(dat_dir_str)
        img_dir = Path(img_dir_str)
        img_dir.mkdir(parents=True, exist_ok=True)

        if not dat_dir.exists():
            print(f"⚠️ 输入目录不存在: {dat_dir}")
            continue

        dat_files = sorted(dat_dir.glob("*.dat"))
        if not dat_files:
            print(f"🔍 在 {dat_dir} 中未找到 .dat 文件")
            continue

        print(f"\n📂 正在处理目录: {dat_dir} → {img_dir}")
        print(f"   找到 {len(dat_files)} 个文件")
        for dat_file in dat_files:
            process_dat(dat_file, img_dir)

    print("\n🎉 所有任务处理完成！")
