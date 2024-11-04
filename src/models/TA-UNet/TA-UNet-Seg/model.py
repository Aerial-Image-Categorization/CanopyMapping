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



import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Gate for Skip Connections
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Modified Up module with Attention Gate and Hybrid Upsampling
class UpWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_attention_gate=True):
        super(UpWithAttention, self).__init__()
        self.use_attention_gate = use_attention_gate

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
        if use_attention_gate:
            self.attention_gate = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Apply attention gate on skip connection
        if self.use_attention_gate:
            x2 = self.attention_gate(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Modified TAUNet with Attention Gates in Skip Connections
class TAUNetSegmentation(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(TAUNetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Bottleneck with positional encoding and self-attention
        self.pos_enc = PositionalEncoding(1024 // factor)
        self.bottleneck = TransformerBlock(1024 // factor, num_heads=8, mlp_dim=2048)

        # Decoder with Attention Gates
        self.up1 = UpWithAttention(1024, 512 // factor, bilinear, use_attention_gate=True)
        self.up2 = UpWithAttention(512, 256 // factor, bilinear, use_attention_gate=True)
        self.up3 = UpWithAttention(256, 128 // factor, bilinear, use_attention_gate=True)
        self.up4 = UpWithAttention(128, 64, bilinear, use_attention_gate=False)
        
        # Output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.pos_enc(self.down4(x4).flatten(2).transpose(0, 1)).transpose(0, 1).view_as(x4)

        # Bottleneck Transformer Block
        x5 = self.bottleneck(x5.flatten(2).transpose(0, 1)).transpose(0, 1).view_as(x5)

        # Decoder with Attention Gate-enabled skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

# This structure allows the TAUNet model to retain critical boundary details and multi-scale features during decoding,
# making it more suited for object segmentation than just localization.



import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    def __init__(self, channels):
        super(CrossAttentionLayer, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, branch_features, canopy_features):
        attention_map = torch.sigmoid(self.conv1(branch_features))
        return canopy_features * attention_map + self.conv2(branch_features)

class PolygonHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(PolygonHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return torch.sigmoid(self.conv2(x))

class TA_UNet_DualBranch(nn.Module):
    def __init__(self, branch_channels=64, canopy_channels=64):
        super(TA_UNet_DualBranch, self).__init__()
        
        # Shared encoder modules
        self.branch_encoder = DoubleConv(3, branch_channels)
        self.canopy_encoder = DoubleConv(3, canopy_channels)
        
        # Cross-attention layer
        self.cross_attention = CrossAttentionLayer(branch_channels)

        # Polygon segmentation head
        self.polygon_head = PolygonHead(branch_channels + canopy_channels)

    def forward(self, x):
        branch_features = self.branch_encoder(x)
        canopy_features = self.canopy_encoder(x)
        
        # Cross-attention to transfer branch info to canopy features
        cross_attention_features = self.cross_attention(branch_features, canopy_features)
        
        # Concatenate features and pass to polygon segmentation head
        combined_features = torch.cat([branch_features, cross_attention_features], dim=1)
        canopy_polygon = self.polygon_head(combined_features)
        
        return canopy_polygon
