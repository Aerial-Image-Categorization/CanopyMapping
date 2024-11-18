import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EncoderBlock(nn.Module):
    """Encoder block with Conv -> Dropout -> Pooling"""
    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    """Decoder block with Conv -> Upsampling"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        
        if x.shape != skip_connection.shape:
            diffY = skip_connection.size()[2] - x.size()[2]
            diffX = skip_connection.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x

class model(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_prob=0.2):
        super(model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.bilinear = True  

        # Encoder
        self.enc1 = EncoderBlock(n_channels, 32, dropout_prob)
        self.enc2 = EncoderBlock(32, 64, dropout_prob)
        self.enc3 = EncoderBlock(64, 128, dropout_prob)
        self.enc4 = EncoderBlock(128, 256, dropout_prob)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, 32)

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec1 = self.dec1(bottleneck, enc4)
        dec2 = self.dec2(dec1, enc3)
        dec3 = self.dec3(dec2, enc2)
        dec4 = self.dec4(dec3, enc1)

        output = self.final_conv(dec4)
        output = F.interpolate(output, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)  # Resize to match input
        return output

    def use_checkpointing(self):
        self.enc1 = checkpoint.checkpoint(self.enc1.forward, preserve_rng_state=False)
        self.enc2 = checkpoint.checkpoint(self.enc2.forward, preserve_rng_state=False)
        self.enc3 = checkpoint.checkpoint(self.enc3.forward, preserve_rng_state=False)
        self.enc4 = checkpoint.checkpoint(self.enc4.forward, preserve_rng_state=False)
        self.bottleneck = checkpoint.checkpoint(self.bottleneck.forward, preserve_rng_state=False)
        self.dec1 = checkpoint.checkpoint(self.dec1.forward, preserve_rng_state=False)
        self.dec2 = checkpoint.checkpoint(self.dec2.forward, preserve_rng_state=False)
        self.dec3 = checkpoint.checkpoint(self.dec3.forward, preserve_rng_state=False)
        self.dec4 = checkpoint.checkpoint(self.dec4.forward, preserve_rng_state=False)
        self.final_conv = checkpoint.checkpoint(self.final_conv.forward, preserve_rng_state=False)

    def save_pretrained(self, save_directory):
        model_path = f"{save_directory}/pytorch_model.bin"
        torch.save(self.state_dict(), model_path)
        # Assuming config_class is defined elsewhere for saving configuration
        # self.config_class.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device, *model_args, **kwargs):
        # Assuming config is defined elsewhere for loading configuration
        # unetconfig = config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(*model_args, **kwargs)
        state_dict = torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/pytorch_model.bin", 
            map_location=device
        )
        model.load_state_dict(state_dict)
        return model