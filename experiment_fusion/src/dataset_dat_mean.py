# src/dataset_dat_mean.py
# Mean Centering 版本的多模态数据集
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from data_utils_dat_mean import load_timeseries, normalize_per_series

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, split='train', use_last_n=2000, seed=42):
        self.use_last_n = use_last_n
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        samples = []
        img_dir = os.path.join(os.path.dirname(root_dir), 'img_new')
        # 收敛数据
        for fname in os.listdir(os.path.join(root_dir, 'dat_converge')):
            if fname.endswith('.dat'):
                base = fname[:-4]
                img_path = os.path.join(img_dir, 'img_converge', f"{base}.png")
                dat_path = os.path.join(root_dir, 'dat_converge', fname)
                if os.path.exists(img_path):
                    samples.append((dat_path, img_path, 1))
        # 未收敛数据
        for fname in os.listdir(os.path.join(root_dir, 'dat_un-converge')):
            if fname.endswith('.dat'):
                base = fname[:-4]
                img_path = os.path.join(img_dir, 'img_un-converge', f"{base}.png")
                dat_path = os.path.join(root_dir, 'dat_un-converge', fname)
                if os.path.exists(img_path):
                    samples.append((dat_path, img_path, 0))

        # 随机划分
        np.random.seed(seed)
        idx = np.random.permutation(len(samples))
        n_train = int(0.64 * len(idx))
        n_val = int(0.16 * len(idx))
        if split == 'train':
            self.samples = [samples[i] for i in idx[:n_train]]
        elif split == 'val':
            self.samples = [samples[i] for i in idx[n_train:n_train+n_val]]
        else:
            self.samples = [samples[i] for i in idx[n_train+n_val:]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dat_path, img_path, label = self.samples[idx]
        ts = load_timeseries(dat_path, self.use_last_n)
        ts = normalize_per_series(ts)  # Mean Centering
        ts = torch.from_numpy(ts).float()
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, ts, torch.tensor(label, dtype=torch.long)
