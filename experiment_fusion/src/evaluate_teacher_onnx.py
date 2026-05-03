# src/evaluate_teacher_onnx.py
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset import MultimodalDataset
import onnxruntime as ort

def main():
    print("🚀 开始评估 ONNX 教师模型性能...")
    
    # 1. 加载测试集
    # 文档要求：测试所有数据集，不跳过
    test_dataset = MultimodalDataset("../dat", split='test')
    # 关键：batch_size=1 以便精确计算单样本推理时间，符合 <100ms 的指标要求
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"📊 测试集样本数：{len(test_dataset)}")

    # 2. 加载 ONNX 模型
    onnx_path = "../dataset/teacher_multimodal.onnx"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"❌ ONNX 模型不存在：{onnx_path}\n   请先运行 export_teacher_onnx.py")
    
    # 配置执行提供者
    # 当前环境仅支持 CPU，若后续有 GPU 可改为 ['CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = ['CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
        print(f"✅ ONNX 教师模型加载成功 (设备：{providers[0]})")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    # 3. 逐样本推理与计时
    all_preds = []
    all_labels = []
    inference_times = []

    print("⏳ 正在逐样本推理并记录耗时...")
    
    for i, (img, ts, label) in enumerate(test_loader):
        # 转为 numpy (batch_size=1)
        img_np = img.numpy()      # (1, 3, 224, 224)
        ts_np = ts.numpy()        # (1, 2000, 2) - 对应 seq_len=2000
        
        # 维度检查 (可选，确保与训练一致)
        assert img_np.shape == (1, 3, 224, 224), f"图像维度错误：{img_np.shape}"
        assert ts_np.shape[1] == 2000, f"时序长度错误：{ts_np.shape[1]}，应为 2000"

        start_time = time.perf_counter()
        
        try:
            # ONNX 推理
            outputs = sess.run(None, {"image": img_np, "timeseries": ts_np})
            logits = outputs[0]       # (1, 2)
            pred = np.argmax(logits, axis=1)[0]
        except Exception as e:
            print(f"⚠️ 第 {i} 个样本推理失败：{e}")
            continue
            
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000  # 转换为毫秒
        inference_times.append(elapsed_ms)
        
        all_preds.append(pred)
        all_labels.append(label.item())

    if len(all_preds) == 0:
        print("❌ 没有成功推理任何样本，程序终止。")
        return

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # 4. 计算性能指标
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    
    # 解析混淆矩阵: [[TN, FP], [FN, TP]] 
    # 假设 0=Unconverged, 1=Converged
    tn, fp, fn, tp = cm.ravel()
    
    recall_converged = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_unconverged = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 推理时间统计
    avg_time = np.mean(inference_times)
    max_time = np.max(inference_times)
    min_time = np.min(inference_times)
    pass_100ms = (max_time < 100.0)

    # 5. 输出结果
    print("\n" + "="*60)
    print("🎯 教师模型 ONNX 测试集性能报告")
    print("="*60)
    print(f"总体准确率 (Accuracy):       {acc:.4f} ({acc*100:.2f}%)")
    print(f"收敛数据召回率 (Recall↑):   {recall_converged:.4f} ({recall_converged*100:.2f}%)")
    print(f"未收敛数据召回率 (Recall↓): {recall_unconverged:.4f} ({recall_unconverged*100:.2f}%)")
    
    print("\n⏱️ 推理时间性能 (单样本):")
    print(f"   平均耗时：{avg_time:.2f} ms")
    print(f"   最大耗时：{max_time:.2f} ms")
    print(f"   最小耗时：{min_time:.2f} ms")
    if pass_100ms:
        print(f"   ✅ 满足实时性要求 (< 100ms)")
    else:
        print(f"   ⚠️ 警告：最大耗时超过 100ms 限制！")

    print("\n🧮 混淆矩阵 (Actual \\ Predicted):")
    print(f"             Pred-Unconv  Pred-Conv")
    print(f"Act-Unconv     {tn:5d}       {fp:5d}")
    print(f"Act-Conv       {fn:5d}       {tp:5d}")
    
    print("\n📋 详细分类报告:")
    # target_names 根据文档习惯：0=未收敛，1=收敛
    print(classification_report(labels, preds, target_names=["Unconverged", "Converged"], digits=4))

if __name__ == "__main__":
    main()