import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, heads=8, head_dim=16):
        super(AttentionLayer, self).__init__()
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(input_dim, heads * head_dim * 3, bias=False)
        self.to_out = nn.Linear(heads * head_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = q * self.scale
        dots = torch.einsum('bhid,bhjd->bhij', q, k)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out).squeeze(1)

class BitDiffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BitDiffusion, self).__init__()
        self.input_dim = input_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            AttentionLayer(hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)