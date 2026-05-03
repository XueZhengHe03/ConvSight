# split_dataset.py
import os
import shutil
import random
from pathlib import Path

def split_images(src_dir, train_dir, val_dir, test_dir, ratios=(0.64, 0.16, 0.2)):
    """
    将 src_dir 中的图片按比例划分为 train/val/test
    ratios: (train_ratio, val_ratio, test_ratio)
    """
    image_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    if not image_files:
        print(f"⚠️ {src_dir} 中没有 PNG 图片")
        return

    random.shuffle(image_files)
    n = len(image_files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # n_test = n - n_train - n_val

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    # 复制文件
    for f in train_files:
        shutil.copy(os.path.join(src_dir, f), train_dir)
    for f in val_files:
        shutil.copy(os.path.join(src_dir, f), val_dir)
    for f in test_files:
        shutil.copy(os.path.join(src_dir, f), test_dir)

    print(f"✅ {src_dir} 分割完成:")
    print(f"   训练: {len(train_files)} | 验证: {len(val_files)} | 测试: {len(test_files)}")

# 创建目标目录
os.makedirs('../dataset/train/converge', exist_ok=True)
os.makedirs('../dataset/train/unconverge', exist_ok=True)
os.makedirs('../dataset/val/converge', exist_ok=True)
os.makedirs('../dataset/val/unconverge', exist_ok=True)
os.makedirs('../dataset/test/converge', exist_ok=True)
os.makedirs('../dataset/test/unconverge', exist_ok=True)

# 执行分割
split_images(
    "../dat/img_converge",
    "../dataset/train/converge",
    "../dataset/val/converge",
    "../dataset/test/converge"
)

split_images(
    "../dat/img_un-converge",
    "../dataset/train/unconverge",
    "../dataset/val/unconverge",
    "../dataset/test/unconverge"
)

print("\n🎉 数据集划分完成！结构如下：")
print("dataset/")
print("├── train/     ← 用于训练")
print("├── val/       ← 用于调参和选模型")
print("└── test/      ← 用于最终评估（只用一次！）")