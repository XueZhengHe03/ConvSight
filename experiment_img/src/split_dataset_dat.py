# split_dataset_dat.py
import os
import shutil
import random
from pathlib import Path

def split_images(src_dir, train_dir, val_dir, test_dir, ratios=(0.64, 0.16, 0.2)):
    """将 src_dir 中的图片按比例划分为 train/val/test"""
    image_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    if not image_files:
        print(f"⚠️ {src_dir} 中没有 PNG 图片")
        return

    random.shuffle(image_files)
    n = len(image_files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

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
os.makedirs('../dataset_new/train/converge', exist_ok=True)
os.makedirs('../dataset_new/train/unconverge', exist_ok=True)
os.makedirs('../dataset_new/val/converge', exist_ok=True)
os.makedirs('../dataset_new/val/unconverge', exist_ok=True)
os.makedirs('../dataset_new/test/converge', exist_ok=True)
os.makedirs('../dataset_new/test/unconverge', exist_ok=True)

# 执行分割
split_images(
    "../img_new/img_converge",
    "../dataset_new/train/converge",
    "../dataset_new/val/converge",
    "../dataset_new/test/converge"
)

split_images(
    "../img_new/img_un-converge",
    "../dataset_new/train/unconverge",
    "../dataset_new/val/unconverge",
    "../dataset_new/test/unconverge"
)

print("\n🎉 数据集划分完成！")
