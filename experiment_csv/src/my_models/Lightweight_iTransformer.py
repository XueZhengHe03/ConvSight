# src/models/Lightweight_iTransformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        E = self.d_head
        
        q = self.q_proj(x).view(B, L, H, E)
        k = self.k_proj(x).view(B, L, H, E)
        v = self.v_proj(x).view(B, L, H, E)
        
        scores = torch.einsum("blhe,bshe->bhls", q, k) / (E ** 0.5)
        A = torch.softmax(scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, v)
        output = V.reshape(B, L, D)
        return self.out_proj(output)

class LightweightEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = LightweightAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        y = self.norm2(x)
        y = self.dropout(F.relu(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return x + y

class Lightweight_iTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = configs.num_class
        self.d_model = configs.d_model
        
        # 轻量化嵌入
        self.token_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model))
        self.dropout = nn.Dropout(configs.dropout)
        
        # 单层编码器
        self.encoder = LightweightEncoderLayer(
            configs.d_model, 
            configs.n_heads, 
            configs.d_ff, 
            configs.dropout
        )
        
        # 分类头
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc):
        # 嵌入
        enc_out = self.token_embedding(x_enc) + self.position_embedding[:, :x_enc.size(1)]
        enc_out = self.dropout(enc_out)
        
        # 单层编码
        enc_out = self.encoder(enc_out)
        
        # 分类
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)