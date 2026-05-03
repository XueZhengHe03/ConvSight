# src/benchmark_teacher_models.py
import os
import time
import numpy as np
import torch
from model import MultimodalFusionModel
import onnxruntime as ort
from torchvision import transforms
from PIL import Image

def load_test_sample():
    """加载一个测试样本（图像 + 时序）"""
    # 图像
    img_path = "../dat/img_converge/001.png"
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    # 时序
    import pandas as pd
    df = pd.read_csv("../dat/csv_converge/001.csv")
    v2 = df['v2'].values.astype(np.float32)
    v3 = df['v3'].values.astype(np.float32)
    ts = np.stack([v2, v3], axis=1)
    if len(ts) < 2000:
        pad = np.tile(ts[0], (2000 - len(ts), 1))
        ts = np.vstack([pad, ts])
    else:
        ts = ts[-2000:]
    mean = ts.mean(axis=0)
    std = ts.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    ts = (ts - mean) / std
    ts_tensor = torch.from_numpy(ts).float().unsqueeze(0)  # (1, 2000, 2)

    return img_tensor, ts_tensor

def benchmark_pytorch(model, img, ts, num_runs=100):
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(img, ts)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start = time.time()
        for _ in range(num_runs):
            _ = model(img, ts)
        end = time.time()
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms

def benchmark_onnx(onnx_path, img, ts, num_runs=100):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    img_np = img.numpy()
    ts_np = ts.numpy()

    # 预热
    for _ in range(10):
        _ = sess.run(None, {"image": img_np, "timeseries": ts_np})

    start = time.time()
    for _ in range(num_runs):
        _ = sess.run(None, {"image": img_np, "timeseries": ts_np})
    end = time.time()
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms

def main():
    print("🚀 教师模型推理时间对比 (CPU)")
    print("=" * 50)

    # 加载数据
    img, ts = load_test_sample()
    img_cpu = img.cpu()
    ts_cpu = ts.cpu()

    # PyTorch 模型
    model = MultimodalFusionModel(seq_len=2000, d_model=128)
    model.load_state_dict(torch.load("../dataset/best_multimodal.pth", map_location='cpu'))
    model.eval()

    pytorch_time = benchmark_pytorch(model, img_cpu, ts_cpu, num_runs=100)
    print(f"PyTorch 教师模型: {pytorch_time:.2f} ms")

    # ONNX 模型
    onnx_path = "../dataset/teacher_multimodal.onnx"
    if not os.path.exists(onnx_path):
        print("❌ ONNX 模型未找到，请先运行 export_teacher_onnx.py")
        return

    onnx_time = benchmark_onnx(onnx_path, img_cpu, ts_cpu, num_runs=100)
    print(f"ONNX 教师模型:   {onnx_time:.2f} ms")

    # 加速比
    speedup = pytorch_time / onnx_time
    print(f"\n✅ ONNX 加速比: {speedup:.2f}x")
    print(f"✅ 满足 <100ms 要求: {'是' if onnx_time < 100 else '否'}")

if __name__ == "__main__":
    main()