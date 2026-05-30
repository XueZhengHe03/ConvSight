# src/cross_domain_test.py
# 跨域泛化实验：用CSV训练的模型在DAT数据上测试
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from model_new import MultimodalFusionModel

def load_dat_as_2channel(dat_path, use_last_n=2000):
    """加载DAT数据并转为2通道格式"""
    data = np.loadtxt(dat_path, skiprows=4)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    seq = data[:, 1:2].astype(np.float32)
    # 复制为2通道
    seq_2ch = np.concatenate([seq, seq], axis=1)
    if len(seq_2ch) < use_last_n:
        pad = np.tile(seq_2ch[0], (use_last_n - len(seq_2ch), 1))
        seq_2ch = np.vstack([pad, seq_2ch])
    else:
        seq_2ch = seq_2ch[-use_last_n:]
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(seq_2ch).astype(np.float32)

def main():
    device = torch.device("cpu")

    # 加载CSV训练的模型 (Dynamic Gated Fusion, d_model=64)
    model = MultimodalFusionModel(seq_len=2000, d_model=64, num_classes=2)
    model_path = "../dataset/best_multimodal_new.pth"

    if not os.path.exists(model_path):
        print("❌ CSV训练的模型不存在，请先运行 train_multimodal.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 加载CSV训练的模型: {model_path}")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DAT数据路径
    dat_root = "../dat_new"
    img_root = "../img_new"

    # 测试结果
    all_preds = []
    all_labels = []
    results = []

    print("\n📊 跨域泛化测试结果:")
    print("-" * 60)

    for category, label in [('dat_converge', 1), ('dat_un-converge', 0)]:
        dat_dir = os.path.join(dat_root, category)
        img_dir = os.path.join(img_root, category.replace('dat_', 'img_'))

        correct = 0
        total = 0

        for fname in sorted(os.listdir(dat_dir)):
            if fname.endswith('.dat'):
                dat_path = os.path.join(dat_dir, fname)
                img_path = os.path.join(img_dir, fname.replace('.dat', '.png'))

                if not os.path.exists(img_path):
                    continue

                # 加载时序数据 (2通道)
                ts = load_dat_as_2channel(dat_path)
                ts_tensor = torch.from_numpy(ts).unsqueeze(0).float()  # (1, T, 2)

                # 加载图像
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

                # 预测
                with torch.no_grad():
                    logits = model(img_tensor, ts_tensor)
                    pred = logits.argmax(dim=1).item()

                all_preds.append(pred)
                all_labels.append(label)
                total += 1
                if pred == label:
                    correct += 1

                # 显示每个样本的结果
                status = "✅" if pred == label else "❌"
                results.append(f"{status} {fname}: 预测={pred}, 真实={label}")

        acc = correct / total if total > 0 else 0
        print(f"\n{category}: {correct}/{total} ({acc:.2%})")

    # 打印详细结果
    print("\n" + "=" * 60)
    print("详细预测结果:")
    print("=" * 60)
    for r in results:
        print(r)

    # 总体统计
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total_acc = (all_preds == all_labels).mean()
    conv_acc = (all_preds[all_labels == 1] == 1).mean() if sum(all_labels == 1) > 0 else 0
    unconv_acc = (all_preds[all_labels == 0] == 0).mean() if sum(all_labels == 0) > 0 else 0

    print("\n" + "=" * 60)
    print("📈 跨域泛化实验总结:")
    print("=" * 60)
    print(f"总体准确率:     {total_acc:.2%}")
    print(f"收敛样本召回率: {conv_acc:.2%}")
    print(f"未收敛样本召回率: {unconv_acc:.2%}")
    print(f"\n注: 此模型在CSV数据集上训练，直接在DAT数据集上测试")

if __name__ == "__main__":
    main()
