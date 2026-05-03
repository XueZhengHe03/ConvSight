# src/export_teacher_onnx.py
import torch
import onnx
from model import MultimodalFusionModel

def main():
    # --- 修改 1: 模型参数与训练代码完全一致 ---
    seq_len = 2000
    d_model = 32          # 与 train_multimodal.py 中的 d_model=32 一致
    num_classes = 2       # 与训练代码一致
    
    # 加载教师模型（与训练时完全一致）
    model = MultimodalFusionModel(seq_len=seq_len, d_model=d_model, num_classes=num_classes)
    
    # --- 修改 2: 模型文件路径与训练代码保存路径一致 ---
    ckpt_path = '../dataset/best_multimodal.pth'
    
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ 模型加载成功：{ckpt_path}")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print(f"   请确认文件路径是否正确：{ckpt_path}")
        return

    model.eval()

    # 创建符合输入要求的 dummy 数据
    # --- 修改 3: 确认图像输入维度与训练时一致 ---
    dummy_img = torch.randn(1, 3, 224, 224)   # 图像输入 [B, C, H, W]
    dummy_ts = torch.randn(1, seq_len, 2)     # 时序输入 [B, Seq_Len, Features]

    # 导出 ONNX
    try:
        torch.onnx.export(
            model,
            (dummy_img, dummy_ts),
            "../dataset/teacher_multimodal.onnx",
            input_names=["image", "timeseries"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch"},
                "timeseries": {0: "batch"},
                "logits": {0: "batch"}
            },
            opset_version=13,
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        print("✅ 教师模型 ONNX 已导出：../dataset/teacher_multimodal.onnx")

        # 验证模型
        onnx_model = onnx.load("../dataset/teacher_multimodal.onnx")
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX 模型验证通过")
        
        # 打印模型信息
        print(f"\n📋 模型信息:")
        print(f"   - 输入图像维度：{list(dummy_img.shape)}")
        print(f"   - 输入时序维度：{list(dummy_ts.shape)}")
        print(f"   - 输出维度：{list(model(dummy_img, dummy_ts).shape)}")

    except Exception as e:
        print(f"❌ ONNX 导出失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()