import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicThresholdDenoising(nn.Module):
    def __init__(self, input_dim):
        super(DynamicThresholdDenoising, self).__init__()
        self.threshold = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        return torch.where(torch.abs(x) > self.threshold, x, torch.zeros_like(x))

class SwinTransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SwinTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [seq_len, batch_size, input_dim]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        return x

class DRSwinNet(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes):
        super(DRSwinNet, self).__init__()
        self.denoising = DynamicThresholdDenoising(input_dim)
        self.transformer = SwinTransformerBlock(input_dim, num_heads)
        self.classifier = nn.Linear(input_dim, num_classes)



    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += H * W * self.dim
        # W-MSA/SW-MSA
        nW = H * W // self.window_size // self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += H * W * self.dim * self.mlp_ratio * 2
        flops += H * W * self.dim
        return flops
