# src/model.py
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import importlib.util

# --- 添加 Time-Series-Library 路径 ---
ts_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Time-Series-Library'))
if ts_lib_path not in sys.path:
    sys.path.insert(0, ts_lib_path)

for mod_name in ["utils", "layers"]:
    mod_init = os.path.join(ts_lib_path, mod_name, "__init__.py")
    if os.path.exists(mod_init):
        spec = importlib.util.spec_from_file_location(mod_name, mod_init)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

from models.iTransformer import Model as iTransformer

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=2, seq_len=2000, d_model=32): # [修改 1] d_model 降至 32
        super().__init__()
        # 图像分支：ResNet50（冻结前 4 个 stage，只训练 fc 部分等效结构）
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # [修改 2] 冻结 layer1, layer2, layer3, layer4，只保留最后分类层可训（或通过微调策略）
        # 这里为了严格限制能力，冻结所有卷积层
        for name, param in resnet.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        img_feat_dim = 2048

        # 时序分支：iTransformer
        enc_in = 2
        config = type('Config', (), {
            'task_name': 'classification',
            'seq_len': seq_len,
            'pred_len': 0,
            'enc_in': enc_in,
            'dec_in': enc_in,
            'num_class': num_classes,
            'd_model': d_model,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 256,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
            'activation': 'gelu',
            'output_attention': False,
            'do_predict': False,
            'factor': 1
        })()
        self.ts_encoder = iTransformer(config)

        # 融合头
        # [修改 3] 时序特征改为 d_model (因为做了 pooling)，不再乘以 seq_len
        ts_feat_dim = d_model 
        fusion_dim = img_feat_dim + ts_feat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128), # [修改 4] 隐藏层从 512 降至 256
            nn.ReLU(),
            nn.Dropout(0.5),           # [修改 5] Dropout 从 0.3 增至 0.45
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_ts):
        B = x_img.size(0)
        img_feat = self.img_encoder(x_img).view(B, -1)
        
        enc_out = self.ts_encoder.enc_embedding(x_ts, None)
        enc_out, _ = self.ts_encoder.encoder(enc_out, attn_mask=None)
        
        # [修改 6] 关键修改：使用全局平均池化，而不是展平所有时间步
        # 原代码：ts_feat = enc_out.reshape(B, -1) -> 维度过大，易过拟合
        ts_feat = enc_out.mean(dim=1) 
        
        fused = torch.cat([img_feat, ts_feat], dim=1)
        logits = self.classifier(fused)
        return logits