# src/model_new.py
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import importlib.util

# --- 添加 Time-Series-Library 路径 (保持不变) ---
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

class DynamicGatedFusion(nn.Module):
    """
    [核心创新] 动态门控融合模块
    功能：根据每个样本的具体特征，动态学习图像和时序特征的权重。
    """
    def __init__(self, img_dim, ts_dim, hidden_dim=384): # [修改 1] hidden_dim 从 512 降至 384，限制融合能力
        super().__init__()
        # 1. 将不同维度的特征投影到同一空间
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.ts_proj = nn.Sequential(
            nn.Linear(ts_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 2. 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 3. 融合后的 BatchNorm
        self.fusion_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, img_feat, ts_feat):
        img_proj = self.img_proj(img_feat)
        ts_proj = self.ts_proj(ts_feat)
        
        combined = torch.cat([img_proj, ts_proj], dim=1)
        alpha = self.gate_network(combined)
        
        fused = alpha * img_proj + (1 - alpha) * ts_proj
        
        return self.fusion_norm(fused)

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=2, seq_len=2000, d_model=64):
        super().__init__()
        # --- 图像分支 (保持不变) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for name, param in resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        img_feat_dim = 2048

        # --- 时序分支 (保持不变) ---
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
            'dropout': 0.3, 
            'embed': 'timeF',
            'freq': 'h',
            'activation': 'gelu',
            'output_attention': False,
            'do_predict': False,
            'factor': 1
        })()
        self.ts_encoder = iTransformer(config)
        
        # 修正时序特征维度计算 (iTransformer 输出为 d_model)
        ts_feat_dim = d_model 

        # --- [核心创新] 动态门控融合 ---
        # 使用修改后的 hidden_dim (384)
        self.fusion = DynamicGatedFusion(img_dim=img_feat_dim, ts_dim=ts_feat_dim, hidden_dim=384)
        
        # --- 分类头 (调整以略微降低拟合能力) ---
        # 融合输出维度现在是 384
        fusion_output_dim = 384 
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),      # [修改 2] 隐藏层从 256 降至 128
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.45),                       # [修改 3] Dropout 从 0.3 增至 0.45，增加正则化
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_ts):
        B = x_img.size(0)
        
        # 1. 提取图像特征
        img_feat = self.img_encoder(x_img).view(B, -1)
        
        # 2. 提取时序特征
        enc_out = self.ts_encoder.enc_embedding(x_ts, None)
        enc_out, _ = self.ts_encoder.encoder(enc_out, attn_mask=None)
        # 全局平均池化
        ts_feat = enc_out.mean(dim=1)
        
        # 3. 动态门控融合
        fused_feat = self.fusion(img_feat, ts_feat)
        # fused_feat shape: [B, 384]
        
        # 4. 分类
        logits = self.classifier(fused_feat)
        return logits