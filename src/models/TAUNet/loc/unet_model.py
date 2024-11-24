import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class HybridAttentionDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with Squeeze-and-Excitation Attention"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(HybridAttentionDoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        #Squeeze-and-Excitation Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.double_conv(x)
        se_out = self.se(conv_out)
        return conv_out * se_out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            HybridAttentionDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class EnhancedUp(nn.Module):
    """Upscaling with Learnable Interpolation and Channel Attention"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(EnhancedUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = HybridAttentionDoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm1(x)
        x = x.permute(1, 0, 2)  # Change to [seq_len, batch_size, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to [batch_size, seq_len, d_model]
        x = self.norm2(x)
        return self.dropout(x)


class config(PretrainedConfig):
    model_type = "unet"

    def __init__(self, n_channels=3, n_classes=2, bilinear=False, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.d_model = 512
        self.nhead = 8
        self.dim_feedforward = 2048
        self.num_layers = 4

class model(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config_class = config
        self.n_channels = config.n_channels
        self.n_classes = config.n_classes
        self.bilinear = config.bilinear
        self.d_model = config.d_model

        self.inc = HybridAttentionDoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.proj = nn.Conv2d(1024 // factor, self.d_model, kernel_size=1)  # Use convolution for projection
        self.pos_enc = PositionalEncoding(self.d_model)
        self.transformer = TransformerBlock(self.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward, num_layers=config.num_layers)

        self.up1 = EnhancedUp(1024, 512 // factor, self.bilinear)
        self.up2 = EnhancedUp(512, 256 // factor, self.bilinear)
        self.up3 = EnhancedUp(256, 128 // factor, self.bilinear)
        self.up4 = EnhancedUp(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        # Encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Positional encoding
        batch, channels, height, width = x5.size()
        x5_proj = self.proj(x5)  #[batch_size, d_model, height, width]
        x5_flat = x5_proj.flatten(2).permute(0, 2, 1)  #[batch_size, height * width, d_model]
        x5_encoded = self.pos_enc(x5_flat)

        # Transformer block
        x5_trans = self.transformer(x5_encoded)  #[batch_size, height * width, d_model]

        # Reshape back to original spatial dimensions
        x5_trans = x5_trans.permute(0, 2, 1).view(batch, self.d_model, height, width)

        # Decoding
        x = self.up1(x5_trans, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def save_pretrained(self, save_directory):
        model_path = f"{save_directory}/pytorch_model.bin"
        torch.save(self.state_dict(), model_path)
        self.config_class.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device, *model_args, **kwargs):
        unetconfig = config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(unetconfig)
        state_dict = torch.hub.load_state_dict_from_url(f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/pytorch_model.bin", map_location=device)
        model.load_state_dict(state_dict)
        return model
