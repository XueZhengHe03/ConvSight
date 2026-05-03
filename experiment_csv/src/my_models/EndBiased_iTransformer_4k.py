# src/my_models/EndBiased_iTransformer_4k.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 动态加载 Time-Series-Library ===
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tsl_path = os.path.join(project_root, "Time-Series-Library")
if tsl_path not in sys.path:
    sys.path.insert(0, tsl_path)

from layers.Embed import TokenEmbedding, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer

class EndBiasedAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, 
                 attention_dropout=0.1, output_attention=False, end_bias=2000):
        super().__init__()
        self.end_bias = end_bias
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / (E ** 0.5)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # 仅当序列长度为 4000 时应用末端增强
        if L == 4000 and S == 4000 and self.end_bias > 0:
            scores[:, :, :, -self.end_bias:] += 1.0
        
        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), A if self.output_attention else None

class EndBiased_iTransformer_4k(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        
        # 嵌入层
        self.token_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        self.position_embedding = PositionalEmbedding(
            d_model=configs.d_model, 
            max_len=configs.seq_len
        )
        self.dropout = nn.Dropout(configs.dropout)
        
        # 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        EndBiasedAttention(
                            False, 
                            configs.factor, 
                            attention_dropout=configs.dropout,
                            output_attention=False,
                            end_bias=2000  # 末端增强步数
                        ),
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # 分类头
        if self.task_name == 'classification':
            self.act = F.gelu
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.token_embedding(x_enc) + self.position_embedding(x_enc)
        enc_out = self.dropout(enc_out)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None