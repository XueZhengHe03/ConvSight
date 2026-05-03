# src/evaluate_model_new_onnx.py
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset import MultimodalDataset
import onnxruntime as ort

def main():
    print("🚀 开始评估 Model_New (Dynamic Gated Fusion) ONNX 性能...")
    
    # 1. 加载测试集
    # 文档要求：测试所有数据集，不跳过
    test_dataset = MultimodalDataset("../dat", split='test')
    # 关键：batch_size=1 以便精确计算单样本推理时间，符合 <100ms 的指标要求
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"📊 测试集样本数：{len(test_dataset)}")

    # 2. 加载 ONNX 模型
    # [修改] 指向新导出的模型文件
    onnx_path = "../dataset/teacher_multimodal_new.onnx"
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"❌ ONNX 模型不存在：{onnx_path}\n"
            f"   请先运行 export_model_new_onnx.py 导出模型。\n"
            f"   注意：确保 model_new.py 的代码结构与训练时完全一致。"
        )
    
    # 配置执行提供者
    # 当前环境仅支持 CPU，若后续有 GPU 可改为 ['CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = ['CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
        print(f"✅ ONNX 模型加载成功 (设备：{providers[0]})")
        
        # 打印输入输出节点信息以供调试
        inputs = [inp.name for inp in sess.get_inputs()]
        outputs = [out.name for out in sess.get_outputs()]
        print(f"   - 输入节点：{inputs}")
        print(f"   - 输出节点：{outputs}")
        
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print("   可能原因：ONNX 文件损坏或算子不支持。")
        return

    # 3. 逐样本推理与计时
    all_preds = []
    all_labels = []
    inference_times = []
    failed_count = 0

    print("⏳ 正在逐样本推理并记录耗时...")
    
    for i, (img, ts, label) in enumerate(test_loader):
        # 转为 numpy (batch_size=1)
        img_np = img.numpy()      # (1, 3, 224, 224)
        ts_np = ts.numpy()        # (1, 2000, 2)
        
        # 维度检查 (确保与 model_new.py 训练配置一致)
        if img_np.shape != (1, 3, 224, 224):
            print(f"⚠️ 第 {i} 个样本图像维度错误：{img_np.shape}，跳过")
            continue
        if ts_np.shape[1] != 2000:
            print(f"⚠️ 第 {i} 个样本时序长度错误：{ts_np.shape[1]}，应为 2000，跳过")
            continue

        start_time = time.perf_counter()
        
        try:
            # ONNX 推理
            # 输入名称必须与 export 时定义的 input_names 一致 ("image", "timeseries")
            outputs = sess.run(None, {"image": img_np, "timeseries": ts_np})
            logits = outputs[0]       # (1, 2)
            pred = np.argmax(logits, axis=1)[0]
        except Exception as e:
            print(f"⚠️ 第 {i} 个样本推理失败：{e}")
            failed_count += 1
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
    
    # 收敛召回率 (Recall for Converged): 真正例 / (真正例 + 假负例)
    # 意义：避免漏判已收敛的计算，节省算力
    recall_converged = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 未收敛召回率 (Recall for Unconverged): 真负例 / (真负例 + 假正例)
    # 意义：避免误判未收敛为收敛，导致结果错误
    recall_unconverged = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 推理时间统计
    avg_time = np.mean(inference_times)
    max_time = np.max(inference_times)
    min_time = np.min(inference_times)
    pass_100ms = (max_time < 100.0)

    # 5. 输出结果
    print("\n" + "="*60)
    print("🎯 Model_New (Dynamic Gated Fusion) ONNX 测试报告")
    print("="*60)
    print(f"总体准确率 (Accuracy):       {acc:.4f} ({acc*100:.2f}%)")
    print(f"收敛数据召回率 (Recall↑):   {recall_converged:.4f} ({recall_converged*100:.2f}%)")
    print(f"未收敛数据召回率 (Recall↓): {recall_unconverged:.4f} ({recall_unconverged*100:.2f}%)")
    
    if failed_count > 0:
        print(f"⚠️ 推理失败样本数：{failed_count}")

    print("\n⏱️ 推理时间性能 (单样本):")
    print(f"   平均耗时：{avg_time:.2f} ms")
    print(f"   最大耗时：{max_time:.2f} ms")
    print(f"   最小耗时：{min_time:.2f} ms")
    
    if pass_100ms:
        print(f"   ✅ 满足实时性要求 (< 100ms)")
    else:
        print(f"   ⚠️ 警告：最大耗时超过 100ms 限制！")
        print(f"      建议：尝试使用 OpenVINO 或 TensorRT 加速，或检查 CPU 负载。")

    print("\n🧮 混淆矩阵 (Actual \\ Predicted):")
    print(f"             Pred-Unconv  Pred-Conv")
    print(f"Act-Unconv     {tn:5d}       {fp:5d}")
    print(f"Act-Conv       {fn:5d}       {tp:5d}")
    
    print("\n📋 详细分类报告:")
    # target_names 根据文档习惯：0=未收敛，1=收敛
    print(classification_report(labels, preds, target_names=["Unconverged", "Converged"], digits=4))
    
    # 简单分析
    print("\n💡 结果分析:")
    if recall_converged < 0.90:
        print("   - 收敛召回率偏低，可能导致大量已收敛计算被误判为未收敛，浪费计算资源。")
    if recall_unconverged < 0.90:
        print("   - 未收敛召回率偏低，可能导致未收敛计算被误判为收敛，产生错误的物理结果。")
    if acc >= 0.95 and pass_100ms:
        print("   - 🎉 模型性能优异，满足高精度 (<5% 误差) 和实时性 (<100ms) 要求！")

if __name__ == "__main__":
    main()