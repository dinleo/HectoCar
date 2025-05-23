import torch
import torch.nn as nn
import torch.nn.functional as F


class DropResidualNorm(nn.Module):
    def __init__(self, dim, dropout=0.1, eps=1e-5):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.norm(hidden_states + input_tensor)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        B, Nq, D = query.shape
        Nc = context.shape[1]

        q = self.q_proj(query).view(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, Nc, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, Nc, self.heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Nq, D)

        return self.out_proj(attn_output)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(mask, -20000)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1, eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(dim, heads, dropout)
        self.self_drn = DropResidualNorm(dim, dropout, eps)

        self.ffn1 = FeedForward(dim, dropout)
        self.ffn1_drn = DropResidualNorm(dim, dropout, eps)

        self.cross_attn = MultiheadCrossAttention(dim, heads, dropout)
        self.cross_drn = DropResidualNorm(dim, dropout, eps)

        self.ffn2 = FeedForward(dim, dropout)
        self.ffn2_drn = DropResidualNorm(dim, dropout, eps)

    def forward(self, query, context, query_mask=None):
        query = self.self_drn(query, self.self_attn(query, query_mask))
        query = self.ffn1_drn(query, self.ffn1(query))
        query = self.cross_drn(query, self.cross_attn(query, context))
        query = self.ffn2_drn(query, self.ffn2(query))
        return query


class HectoEncoder(nn.Module):
    def __init__(self, dim=256, depth=4, heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(dim, heads, dropout) for _ in range(depth)
        ])

    def forward(self, query, context_list, query_mask=None):
        assert len(context_list) == len(self.layers), "Each layer must receive a different context scale"

        for i, layer in enumerate(self.layers):
            query = layer(query, context_list[i], query_mask)
        return query