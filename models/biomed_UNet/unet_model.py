""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from transformers import PreTrainedModel, PretrainedConfig

class config(PretrainedConfig):
    model_type = "unet"
    def __init__(self, n_channels=3, n_classes=2, bilinear=False, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


def load_model(model, path, device):
    state_dict = torch.load(path, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)
    #logging.info(f'Model loaded from {load}')

class model(PreTrainedModel):
    #self.config_class = config
    def __init__(self, config):
        super().__init__(config)
        self.config_class = config
        self.n_channels = config.n_channels
        self.n_classes = config.n_classes
        self.bilinear = config.bilinear
        
        #self.n_channels = n_channels
        #self.n_classes = n_classes
        #self.bilinear = bilinear

        self.inc = (DoubleConv(self.n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, self.bilinear))
        self.up2 = (Up(512, 256 // factor, self.bilinear))
        self.up3 = (Up(256, 128 // factor, self.bilinear))
        self.up4 = (Up(128, 64, self.bilinear))
        self.outc = (OutConv(64, self.n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
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
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        unetconfig = config.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls(unetconfig)
        state_dict = torch.hub.load_state_dict_from_url(f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model