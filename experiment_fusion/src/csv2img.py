import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -----------------------------
# 配置：可添加多个 (输入目录, 输出目录) 对
# -----------------------------
TASKS = [
    ("../dat/csv_converge", "../dat/img_converge"),
    ("../dat/csv_un-converge", "../dat/img_un-converge"),
]

# -----------------------------
# 局部视图参数（可调）
# -----------------------------
WINDOW_SIZE = 2000          # (N - 2000, N)
MIN_X = 1000                # 局部极值计算起始点（文档建议值）
RANGE_MIN_ABS = 0.005       # range 下限绝对值
RANGE_MIN_REL_FACTOR = 0.01 # range 下限相对值比例

# -----------------------------
# 处理单个 CSV 文件
# -----------------------------
def process_csv(csv_path: Path, img_dir: Path):
    try:
        df = pd.read_csv(csv_path)
        if not {'v1', 'v2', 'v3'}.issubset(df.columns):
            print(f"⚠️ 跳过 {csv_path.name}：缺少 v1/v2/v3 列")
            return

        x = df['v1'].values
        v2 = df['v2'].values
        v3 = df['v3'].values
        N = int(x[-1])

        def get_local_ylim(y):
            # 1) 计算 Vavr: (N - WINDOW_SIZE, N)
            start_avg = max(N - WINDOW_SIZE, 1)
            idx_avg = np.searchsorted(x, start_avg, side='left')
            Vavr = np.mean(y[idx_avg:])

            # 2) 动态确定极值计算起始点
            if N <= MIN_X:
                # 若总步数 ≤ 1000，改用后半段（如 N//2 开始），确保有数据
                start_range_val = max(N // 2, 1)
            else:
                start_range_val = MIN_X

            idx_range = np.searchsorted(x, start_range_val, side='left')
            segment = y[idx_range:]

            # 安全兜底：防止空数组（极端情况如 N=1）
            if len(segment) == 0:
                Vmax = Vmin = y[-1]
            else:
                Vmax = np.max(segment)
                Vmin = np.min(segment)

            # 3) 计算 range（使用 abs(Vavr) 避免负值导致 rel 范围为负）
            range_val = max(
                Vmax - Vmin,
                RANGE_MIN_ABS,
                abs(Vavr) * RANGE_MIN_REL_FACTOR
            )
            return Vavr - range_val, Vavr + range_val, Vavr

        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        stem = csv_path.stem  # e.g., "001" 或 "001-un"
        fig.suptitle(f'Convergence Curves - {stem}', fontsize=16, fontweight='bold')

        # 全局 v2
        axes[0, 0].plot(x, v2, color='red', linewidth=1.0)
        axes[0, 0].set_title('Global View - v2')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('v2')
        axes[0, 0].grid(True, linestyle='--', alpha=0.5)

        # 全局 v3
        axes[0, 1].plot(x, v3, color='green', linewidth=1.0)
        axes[0, 1].set_title('Global View - v3')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('v3')
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)

        # 局部 v2
        ymin, ymax, Vavr2 = get_local_ylim(v2)
        axes[1, 0].plot(x, v2, color='purple', linewidth=1.0)
        axes[1, 0].set_xlim(1, N)
        axes[1, 0].set_ylim(ymin, ymax)
        axes[1, 0].set_title(f'Local View - v2\nVavr={Vavr2:.5f}')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('v2')
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)

        # 局部 v3
        ymin, ymax, Vavr3 = get_local_ylim(v3)
        axes[1, 1].plot(x, v3, color='blue', linewidth=1.0)
        axes[1, 1].set_xlim(1, N)
        axes[1, 1].set_ylim(ymin, ymax)
        axes[1, 1].set_title(f'Local View - v3\nVavr={Vavr3:.5f}')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('v3')
        axes[1, 1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_path = img_dir / f"{stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 已生成: {output_path}")

    except Exception as e:
        print(f"❌ 处理 {csv_path.name} 时出错: {e}")

# -----------------------------
# 主程序：遍历所有任务
# -----------------------------
if __name__ == "__main__":
    for csv_dir_str, img_dir_str in TASKS:
        csv_dir = Path(csv_dir_str)
        img_dir = Path(img_dir_str)
        img_dir.mkdir(parents=True, exist_ok=True)

        if not csv_dir.exists():
            print(f"⚠️ 输入目录不存在: {csv_dir}")
            continue

        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            print(f"🔍 在 {csv_dir} 中未找到 .csv 文件")
            continue

        print(f"\n📂 正在处理目录: {csv_dir} → {img_dir}")
        print(f"   找到 {len(csv_files)} 个文件")
        for csv_file in csv_files:
            process_csv(csv_file, img_dir)

    print("\n🎉 所有任务处理完成！")