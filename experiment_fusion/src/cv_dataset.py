# src/cv_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

class CVMultimodalDataset(Dataset):
    def __init__(self, sample_paths, labels, use_last_n=2000):
        self.samples = list(zip(sample_paths, labels))
        self.use_last_n = use_last_n
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, label = self.samples[idx]
        
        # === 加载图像 ===
        base_name = os.path.basename(csv_path)
        name, _ = os.path.splitext(base_name)
        if 'un-converge' in csv_path:
            img_path = os.path.join(os.path.dirname(csv_path).replace('csv_un-converge', 'img_un-converge'), f"{name}.png")
        else:
            img_path = os.path.join(os.path.dirname(csv_path).replace('csv_converge', 'img_converge'), f"{name}.png")
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)  # (3, 224, 224)

        # === 加载时序 ===
        df = pd.read_csv(csv_path)
        v2 = df['v2'].values.astype(np.float32)
        v3 = df['v3'].values.astype(np.float32)
        ts = np.stack([v2, v3], axis=1)  # (T, 2)
        if len(ts) < self.use_last_n:
            pad = np.tile(ts[0], (self.use_last_n - len(ts), 1))
            ts = np.vstack([pad, ts])
        else:
            ts = ts[-self.use_last_n:]
        # 标准化（每条序列独立）
        scaler = StandardScaler()
        ts = scaler.fit_transform(ts).astype(np.float32)
        ts_tensor = torch.from_numpy(ts).float()  # (2000, 2)

        return img_tensor, ts_tensor, torch.tensor(label, dtype=torch.long)