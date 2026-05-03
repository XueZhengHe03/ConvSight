# train_baseline.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights


# -----------------------------
# 数据集类
# -----------------------------
class ConvergenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.images = []
        self.labels = []

        for label_str in ['converge', 'unconverge']:
            label_dir = os.path.join(root_dir, label_str)
            if not os.path.exists(label_dir):
                continue
            for img_name in os.listdir(label_dir):
                if img_name.endswith('.png'):
                    self.images.append(os.path.join(label_dir, img_name))
                    self.labels.append(0 if label_str == 'converge' else 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# -----------------------------
# 模型构建（直接使用官方 pretrained）
# -----------------------------
def build_model(model_name, num_classes=2):
    if model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.92),
            nn.Linear(model.fc.in_features, num_classes)
        )
        return model
    elif model_name == "efficientnet_b0":
        model = timm.create_model('tf_efficientnet_b0', pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_name == "vit_small_patch16_224":
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    elif model_name == "convnext_tiny":
        model = timm.create_model('convnext_tiny', pretrained=True)
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        return model
    elif model_name == "efficientnetv2_s":
        model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_name == "swin_tiny_patch4_window7_224":
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
        # Swin 默认 head 已是 Linear(num_features, num_classes)，但为了安全可重设
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


# -----------------------------
# 训练函数（带进度条）
# -----------------------------
def train_model(model_name, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 训练模型: {model_name}")

    train_dataset = ConvergenceDataset('../dataset/train')
    val_dataset = ConvergenceDataset('../dataset/val')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.95)  # ← 添加 label_smoothing
    optimizer = optim.Adam(model.parameters(), lr=5e-6, weight_decay=5e-4)  # 可调低至 0.0005 若需进一步抑制性能

    best_acc = 0.0
    num_epochs = 10  # ← 关键：只训练 10 轮

    for epoch in range(num_epochs):
        # ========== 训练阶段（带进度条）==========
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 自动处理 Swin/ViT 的 4D 输出（虽然通常不会出现，但保留健壮性）
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{train_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        # ========== 验证阶段 ==========
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if outputs.dim() > 2:
                    outputs = outputs.view(outputs.size(0), -1)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total

        # ========== 保存最佳模型 ==========
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f" ⭐ 保存最佳模型 (验证准确率: {best_acc:.2f}%)")

    print(f"\n✅ {model_name} 训练完成！最佳验证准确率: {best_acc:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name to train')
    args = parser.parse_args()
    model_name = args.model
    save_path = f'../checkpoints/best_{model_name}.pth'
    train_model(model_name, save_path)