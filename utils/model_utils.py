
import torch
import torch.nn as nn

import math
from math import sqrt

class FullAttention(nn.Module):
    """
    attention in model
    """
    def __init__(self, d_model, n_heads, attn_dropout, d_key=None, d_value=None):
        super(FullAttention, self).__init__()
        d_key = d_key or (d_model // n_heads)
        d_value = d_value or (d_model // n_heads)
        self.n_heads = n_heads

        self.WQ = nn.Linear(d_model, d_key*n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_key*n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_value*n_heads, bias=False)
        self.WO = nn.Linear(d_value*n_heads, d_model)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value):
        B, l_query, d_query = query.shape
        _, l_key, _ = key.shape 

        # Matrices projection
        Q = self.WQ(query).view(B, l_query, self.n_heads, -1)
        K = self.WK(key).view(B, l_key, self.n_heads, -1)
        V = self.WV(value).view(B, l_key, self.n_heads, -1)

        # Scaled dot-product attention
        scale = 1./sqrt(Q.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        A = self.dropout(torch.softmax(scale*scores, dim=-1))
        agg = torch.einsum("bhls,bshd->blhd", A, V).contiguous()
        O = self.WO(agg.view(B, l_query, -1))
        return O

class AttnMLP(nn.Module):
    """
    mlp in model
    """
    def __init__(
            self,
            in_dim,
            hidden_dim=None,
            out_dim=None,
            norm_layer=None,
            bias=True,
            drop=0.
    ):
        super(AttnMLP, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias)
        self.act = nn.Sigmoid()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class StepEmbedding(nn.Module):
    def __init__(self, hidden_dim, freq_dim=256):
        super(StepEmbedding, self).__init__()

        """
        Time embedding used in T_phi, and for time embedding
        """

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal_embedding(k, freq_dim, max_period=1000):
        half_dim = freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=k.device)
        k_freqs = k[:, None].float() * freqs[None]
        k_emb = torch.cat([torch.cos(k_freqs), torch.sin(k_freqs)], dim=-1)
        return k_emb 

    def forward(self, k): 
        k_emb = self.sinusoidal_embedding(k, self.freq_dim)
        k_emb = self.mlp(k_emb)
        return k_emb 


class MLPResidual(nn.Module):
    """
    Simple MLP residual network with one hidden state.
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(MLPResidual, self).__init__()
        self.lin_emb = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        self.lin_res = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x_emb = self.lin_emb(x)
        x_res = self.lin_res(x)
        x_out = self.norm(x_emb+x_res)
        return x_out


class DataEmbedding(nn.Module):
    """
    embed for x and y
    """
    def __init__(self, in_dim, out_dim, n_emb):
        super(DataEmbedding, self).__init__()
        self.feat_embedding = [MLPResidual(in_dim, out_dim)]
        if n_emb > 1:
            for i in range(n_emb-1):
                self.feat_embedding.append(MLPResidual(out_dim, out_dim))
        self.feat_embedding = nn.Sequential(*self.feat_embedding)

    def forward(self, x):
        return self.feat_embedding(x)
