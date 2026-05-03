# split_dataset.py
import os
import shutil
import random

def split_data(src_csv_dir, src_img_dir, dst_root, label_name):
    """
    划分单个类别的数据（converge 或 unconverge）
    """
    # 获取公共文件名（去掉扩展名）
    csv_files = {f[:-4] for f in os.listdir(src_csv_dir) if f.endswith('.csv')}
    img_files = {f[:-4] for f in os.listdir(src_img_dir) if f.endswith('.png')}
    common_files = list(csv_files & img_files)
    random.shuffle(common_files)

    n = len(common_files)
    n_train = int(n * 0.64)
    n_val = int(n * 0.16)
    # n_test = n - n_train - n_val

    splits = {
        'train': common_files[:n_train],
        'val': common_files[n_train:n_train+n_val],
        'test': common_files[n_train+n_val:]
    }

    for split, files in splits.items():
        for f in files:
            # 创建目标目录
            csv_dst_dir = os.path.join(dst_root, split, 'csv', label_name)
            img_dst_dir = os.path.join(dst_root, split, 'img', label_name)
            os.makedirs(csv_dst_dir, exist_ok=True)
            os.makedirs(img_dst_dir, exist_ok=True)
            
            # 复制文件
            shutil.copy(
                os.path.join(src_csv_dir, f+'.csv'),
                os.path.join(csv_dst_dir, f+'.csv')
            )
            shutil.copy(
                os.path.join(src_img_dir, f+'.png'),
                os.path.join(img_dst_dir, f+'.png')
            )

# 执行划分
os.makedirs('../dataset', exist_ok=True)
split_data('../dat/csv_converge', '../dat/img_converge', '../dataset', 'converge')
split_data('../dat/csv_un-converge', '../dat/img_un-converge', '../dataset', 'unconverge')

print("✅ 数据集划分完成！")