# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from data_utils import load_timeseries, normalize_per_series

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, split='train', use_last_n=2000, seed=42):
        self.use_last_n = use_last_n
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        samples = []
        # 收敛数据
        for fname in os.listdir(os.path.join(root_dir, 'csv_converge')):
            if fname.endswith('.csv'):
                base = fname[:-4]
                img_path = os.path.join(root_dir, 'img_converge', f"{base}.png")
                csv_path = os.path.join(root_dir, 'csv_converge', fname)
                if os.path.exists(img_path):
                    samples.append((csv_path, img_path, 1))
        # 未收敛数据
        for fname in os.listdir(os.path.join(root_dir, 'csv_un-converge')):
            if fname.endswith('.csv'):
                base = fname[:-4]
                img_path = os.path.join(root_dir, 'img_un-converge', f"{base}.png")
                csv_path = os.path.join(root_dir, 'csv_un-converge', fname)
                if os.path.exists(img_path):
                    samples.append((csv_path, img_path, 0))

        # 随机划分
        np.random.seed(seed)
        idx = np.random.permutation(len(samples))
        n_train = int(0.64 * len(idx))
        n_val = int(0.16 * len(idx))
        if split == 'train':
            self.samples = [samples[i] for i in idx[:n_train]]
        elif split == 'val':
            self.samples = [samples[i] for i in idx[n_train:n_train+n_val]]
        else:  # test
            self.samples = [samples[i] for i in idx[n_train+n_val:]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, img_path, label = self.samples[idx]
        # 时序
        ts = load_timeseries(csv_path, self.use_last_n)
        ts = normalize_per_series(ts)
        ts = torch.from_numpy(ts).float()  # (T, 2)
        # 图像
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)  # (3, 224, 224)
        return img, ts, torch.tensor(label, dtype=torch.long)