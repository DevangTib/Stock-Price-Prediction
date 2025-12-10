import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class TemporalAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.proj = nn.Linear(hid_dim, hid_dim)
        self.ctx = nn.Parameter(torch.randn(hid_dim))

    def forward(self, h):
        # h: (1, seq_len, hid_dim) or (seq_len, hid_dim)
        if h.dim() == 3:
            u = torch.tanh(self.proj(h))               # (1, seq_len, hid)
            attn_scores = u @ self.ctx                 # (1, seq_len)
            alpha = F.softmax(attn_scores, dim=1)      # (1, seq_len)
            return (alpha.unsqueeze(-1) * h).sum(1)    # (1, hid_dim)
        else:
            u = torch.tanh(self.proj(h))               # (seq_len, hid)
            attn_scores = u @ self.ctx                 # (seq_len,)
            alpha = F.softmax(attn_scores, dim=0)      # (seq_len,)
            return (alpha.unsqueeze(-1) * h).sum(0)    # (hid_dim,)

class InformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model) or (seq_len, d_model)
        # unify to batch=1
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False
        B, L, _ = x.size()
        residual = x
        Q = self.q_linear(x).view(B, L, self.n_heads, self.d_k).permute(0,2,1,3)
        K = self.k_linear(x).view(B, L, self.n_heads, self.d_k).permute(0,2,1,3)
        V = self.v_linear(x).view(B, L, self.n_heads, self.d_k).permute(0,2,1,3)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V).permute(0,2,1,3).contiguous().view(B,L,-1)
        out1 = self.norm1(residual + self.dropout(self.out(context)))
        out2 = self.norm2(out1 + self.dropout(self.ff(out1)))
        x = out2
        return x[0] if squeeze_after else x

class TemporalTransformer(nn.Module):
    def __init__(self, d_model, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerLayer(d_model, n_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class PriceEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hid_dim)
        self.pos = nn.Parameter(torch.zeros(100, hid_dim))
        self.transformer = TemporalTransformer(hid_dim, num_layers=2, dropout=dropout)
        self.attn = TemporalAttention(hid_dim)

    def forward(self, x):
        # x: (D,3)
        D = x.size(0)
        last_close = x[-1,0]
        x = x / (last_close + 1e-8)
        x = self.proj(x)  # (D, hid)
        x = x + self.pos[:D]
        x = self.transformer(x)  # (D, hid)
        out = self.attn(x)       # (hid,)
        return out.unsqueeze(0)  # (1, hid)

class TextEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hid_dim)
        self.t_pos = nn.Parameter(torch.zeros(100, hid_dim))
        self.d_pos = nn.Parameter(torch.zeros(30, hid_dim))
        self.t_transformer = TemporalTransformer(hid_dim, num_layers=2, dropout=dropout)
        self.d_transformer = TemporalTransformer(hid_dim, num_layers=2, dropout=dropout)
        self.t_attn = TemporalAttention(hid_dim)
        self.d_attn = TemporalAttention(hid_dim)

    def forward(self, x):
        # x: (D, T, 512)
        day_vecs = []
        D, T, _ = x.size()
        for d in range(D):
            tweets = x[d]
            if tweets.size(0) == 0: continue
            t = self.proj(tweets)       # (T, hid)
            t = t + self.t_pos[:t.size(0)]
            t = self.t_transformer(t)   # (T, hid)
            day_vecs.append(self.t_attn(t))
        if not day_vecs:
            return torch.zeros(1, self.d_pos.size(1), device=x.device)
        d_seq = torch.stack(day_vecs)  # (D', hid)
        d_seq = d_seq + self.d_pos[:d_seq.size(0)]
        d_enc = self.d_transformer(d_seq)  # (D', hid)
        out = self.d_attn(d_enc)           # (hid,)
        return out.unsqueeze(0)

class R_GATModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, stock_num, num_relations):
        super().__init__()
        self.stock_num = stock_num
        self.price_encoders = nn.ModuleList([PriceEncoder(nfeat, nhid) for _ in range(stock_num)])
        self.text_encoders  = nn.ModuleList([TextEncoder(512, nhid) for _ in range(stock_num)])
        self.fuse = nn.ModuleList([nn.Bilinear(nhid, nhid, nhid) for _ in range(stock_num)])
        self.rgc1 = RGCNConv(nhid, nhid, num_relations, num_bases=min(num_relations,30))
        self.rgc2 = RGCNConv(nhid, nhid, num_relations, num_bases=min(num_relations,30))
        self.classifier = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, text, price, edge_index, edge_type):
        feats = []
        for i in range(self.stock_num):
            p = self.price_encoders[i](price[i])  # (1, hid)
            t = self.text_encoders[i](text[i])    # (1, hid)
            feats.append(F.relu(self.fuse[i](p, t)).squeeze(0))
        H = torch.stack(feats)  # (S, hid)
        X = F.dropout(H, p=self.dropout, training=self.training)
        X = F.relu(self.rgc1(X, edge_index, edge_type))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = F.relu(self.rgc2(X, edge_index, edge_type))
        return self.classifier(X)  

# train_gpt_fixed.py
