import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import logging

# Utility function for positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x

# Self-attention block with MLP
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

# Define the cross-attention mechanism for encoder-decoder interaction
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross_input):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(cross_input), self.norm1(cross_input))[0])
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

# Basic U-Net parts with transformer-based modifications
class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_cross_attention=False):
        super(Up, self).__init__()
        self.use_cross_attention = use_cross_attention
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        if use_cross_attention:
            self.cross_attention = CrossAttentionBlock(out_channels, num_heads=4, mlp_dim=out_channels * 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        if self.use_cross_attention:
            x1 = self.cross_attention(x1.flatten(2).transpose(0, 1), x2.flatten(2).transpose(0, 1))
            x1 = x1.transpose(0, 1).view_as(x2)
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TAUNetConfig(PretrainedConfig):
    model_type = "ta_unet"

    def __init__(self, n_channels=3, n_classes=2, bilinear=False, **kwargs):
        super(TAUNetConfig, self).__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


class TAUNet(PreTrainedModel):
    config_class = TAUNetConfig

    def __init__(self, config):
        super(TAUNet, self).__init__(config)
        self.n_channels = config.n_channels
        self.n_classes = config.n_classes
        self.bilinear = config.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.pos_enc = PositionalEncoding(512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Transformer bottleneck
        self.bottleneck = TransformerBlock(1024 // factor, num_heads=8, mlp_dim=2048)

        # Decoder with Cross-Attention
        self.up1 = Up(1024, 512 // factor, self.bilinear, use_cross_attention=True)
        self.up2 = Up(512, 256 // factor, self.bilinear, use_cross_attention=True)
        self.up3 = Up(256, 128 // factor, self.bilinear, use_cross_attention=True)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.pos_enc(self.down4(x4).flatten(2).transpose(0, 1)).transpose(0, 1).view_as(x4)

        # Bottleneck Transformer Block
        x5 = self.bottleneck(x5.flatten(2).transpose(0, 1)).transpose(0, 1).view_as(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
