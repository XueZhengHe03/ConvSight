# src/prepare_cv_data.py
import os

def get_all_samples():
    """
    扫描 dat/ 目录，返回所有 CSV 文件路径和对应标签
    - csv_converge/ → label=1
    - csv_un-converge/ → label=0
    """
    samples = []
    labels = []

    # 收敛数据
    converge_dir = "../dat/csv_converge"
    if os.path.exists(converge_dir):
        for fname in sorted(os.listdir(converge_dir)):
            if fname.endswith('.csv'):
                samples.append(os.path.join(converge_dir, fname))
                labels.append(1)

    # 未收敛数据
    unconverge_dir = "../dat/csv_un-converge"
    if os.path.exists(unconverge_dir):
        for fname in sorted(os.listdir(unconverge_dir)):
            if fname.endswith('.csv'):
                samples.append(os.path.join(unconverge_dir, fname))
                labels.append(0)

    assert len(samples) == len(labels), f"样本数 ({len(samples)}) 与标签数 ({len(labels)}) 不一致！"
    print(f"✅ 共加载 {len(samples)} 个样本（收敛: {sum(labels)}, 未收敛: {len(labels)-sum(labels)}）")
    return samples, labels