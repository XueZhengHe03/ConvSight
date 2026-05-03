# src/export_model_new_onnx.py
import torch
import onnx
from model_new import MultimodalFusionModel
import os

def main():
    print("🚀 开始导出 Model_New (Dynamic Gated Fusion) 到 ONNX...")

    # 1. 配置模型参数 (必须与训练时 train_multimodal.py 完全一致)
    seq_len = 2000
    d_model = 32          # 关键：训练时使用的 d_model
    num_classes = 2
    
    # 2. 实例化模型
    model = MultimodalFusionModel(seq_len=seq_len, d_model=d_model, num_classes=num_classes)
    
    # 3. 加载权重
    ckpt_path = '../dataset/best_multimodal_new.pth'
    if not os.path.exists(ckpt_path):
        # 尝试备用路径
        alt_path = '../dataset/best_multimodal.pth'
        if os.path.exists(alt_path):
            print(f"⚠️ 未找到 {ckpt_path}，尝试使用备用路径：{alt_path}")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"❌ 模型权重文件不存在：{ckpt_path}\n   请先运行 train_multimodal.py 生成权重文件。")

    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ 模型权重加载成功：{ckpt_path}")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print("   可能原因：model_new.py 的代码结构与训练保存时不一致。")
        print("   请检查 DynamicGatedFusion 的 hidden_dim 和 classifier 的层数定义。")
        return

    model.eval()

    # 4. 创建 Dummy 输入 (模拟推理时的输入形状)
    # 图像：[Batch, Channels, Height, Width]
    dummy_img = torch.randn(1, 3, 224, 224)
    # 时序：[Batch, Seq_Len, Features] -> (1, 2000, 2)
    dummy_ts = torch.randn(1, seq_len, 2)

    # 5. 导出 ONNX
    output_path = "../dataset/teacher_multimodal_new.onnx"
    
    try:
        torch.onnx.export(
            model,
            (dummy_img, dummy_ts),
            output_path,
            input_names=["image", "timeseries"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "timeseries": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=13,       # 推荐使用 13 或更高，兼容性好
            export_params=True,     # 存储训练好的权重
            do_constant_folding=True, # 优化常量计算
            verbose=False           # 设置为 True 可查看详细导出日志
        )
        print(f"✅ ONNX 模型导出成功：{output_path}")

        # 6. 验证模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型结构验证通过")

        # 7. 打印模型信息
        print("\n📋 模型详细信息:")
        print(f"   - 输入图像维度：{list(dummy_img.shape)}")
        print(f"   - 输入时序维度：{list(dummy_ts.shape)}")
        
        # 简单测试一次推理
        with torch.no_grad():
            test_out = model(dummy_img, dummy_ts)
        print(f"   - 输出 Logits 维度：{list(test_out.shape)}")
        print(f"   - 参数量：{sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"❌ ONNX 导出或验证失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()