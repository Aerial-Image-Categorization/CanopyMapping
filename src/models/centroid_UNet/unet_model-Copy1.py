import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = False
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_pool1 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(64)
        self.enc_pool2 = nn.MaxPool2d(2)

        self.enc_conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_bn5 = nn.BatchNorm2d(128)
        self.enc_conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_bn6 = nn.BatchNorm2d(128)
        self.enc_pool3 = nn.MaxPool2d(2)

        # Middle Part
        self.middle_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.middle_bn1 = nn.BatchNorm2d(256)
        self.middle_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.middle_bn2 = nn.BatchNorm2d(256)

        # Decoder
        self.dec_upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv7 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.dec_bn7 = nn.BatchNorm2d(128)
        self.dec_conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_bn8 = nn.BatchNorm2d(128)

        self.dec_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv9 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.dec_bn9 = nn.BatchNorm2d(64)
        self.dec_conv10 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_bn10 = nn.BatchNorm2d(64)

        self.dec_upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv11 = nn.Conv2d(64 + 32, 32, 3, padding=1)
        self.dec_bn11 = nn.BatchNorm2d(32)
        self.dec_conv12 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_bn12 = nn.BatchNorm2d(32)

        self.dec_upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv13 = nn.Conv2d(32 + 32, 32, 3, padding=1)
        self.dec_bn13 = nn.BatchNorm2d(32)
        self.dec_conv14 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_bn14 = nn.BatchNorm2d(32)

        # Last Part
        self.last_conv1 = nn.Conv2d(32, 32, 1)
        self.last_bn1 = nn.BatchNorm2d(32)
        self.last_conv2 = nn.Conv2d(32, 32, 1)
        self.last_bn2 = nn.BatchNorm2d(32)
        self.last_conv3 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        e_cnn1 = F.relu(self.enc_bn1(self.enc_conv1(x)))
        e_cnn2 = F.relu(self.enc_bn2(self.enc_conv2(e_cnn1)))
        e_skip1 = F.dropout(e_cnn2, p=0.2)
        e_pool1 = self.enc_pool1(e_skip1)

        e_cnn3 = F.relu(self.enc_bn3(self.enc_conv3(e_pool1)))
        e_cnn4 = F.relu(self.enc_bn4(self.enc_conv4(e_cnn3)))
        e_skip2 = F.dropout(e_cnn4, p=0.2)
        e_pool2 = self.enc_pool2(e_skip2)

        e_cnn5 = F.relu(self.enc_bn5(self.enc_conv5(e_pool2)))
        e_cnn6 = F.relu(self.enc_bn6(self.enc_conv6(e_cnn5)))
        e_skip3 = F.dropout(e_cnn6, p=0.2)
        e_pool3 = self.enc_pool3(e_skip3)

        # Middle Part
        middle_cnn1 = F.relu(self.middle_bn1(self.middle_conv1(e_pool3)))
        middle_cnn2 = F.relu(self.middle_bn2(self.middle_conv2(middle_cnn1)))

        # Decoder
        d_upsample1 = self.dec_upsample1(middle_cnn2)
        d_concat1 = torch.cat((d_upsample1, e_skip3), dim=1)
        d_cnn7 = F.relu(self.dec_bn7(self.dec_conv7(d_concat1)))
        d_cnn8 = F.relu(self.dec_bn8(self.dec_conv8(d_cnn7)))

        d_upsample2 = self.dec_upsample2(d_cnn8)
        d_concat2 = torch.cat((d_upsample2, e_skip2), dim=1)
        d_cnn9 = F.relu(self.dec_bn9(self.dec_conv9(d_concat2)))
        d_cnn10 = F.relu(self.dec_bn10(self.dec_conv10(d_cnn9)))

        d_upsample3 = self.dec_upsample3(d_cnn10)
        d_concat3 = torch.cat((d_upsample3, e_skip1), dim=1)
        d_cnn11 = F.relu(self.dec_bn11(self.dec_conv11(d_concat3)))
        d_cnn12 = F.relu(self.dec_bn12(self.dec_conv12(d_cnn11)))

        d_upsample4 = self.dec_upsample4(d_cnn12)
        d_concat4 = torch.cat((d_upsample4, e_cnn1), dim=1)
        d_cnn13 = F.relu(self.dec_bn13(self.dec_conv13(d_concat4)))
        d_cnn14 = F.relu(self.dec_bn14(self.dec_conv14(d_cnn13)))

        # Last Part
        l_cnn1 = F.relu(self.last_bn1(self.last_conv1(d_cnn14)))
        l_cnn2 = F.relu(self.last_bn2(self.last_conv2(l_cnn1)))
        output = torch.sigmoid(self.last_conv3(l_cnn2))

        return output

    def use_checkpointing(self):
        self.enc_conv1 = torch.utils.checkpoint.checkpoint(self.enc_conv1)
        self.enc_conv2 = torch.utils.checkpoint.checkpoint(self.enc_conv2)
        self.enc_conv3 = torch.utils.checkpoint.checkpoint(self.enc_conv3)
        self.enc_conv4 = torch.utils.checkpoint.checkpoint(self.enc_conv4)
        self.enc_conv5 = torch.utils.checkpoint.checkpoint(self.enc_conv5)
        self.enc_conv6 = torch.utils.checkpoint.checkpoint(self.enc_conv6)
        self.middle_conv1 = torch.utils.checkpoint.checkpoint(self.middle_conv1)
        self.middle_conv2 = torch.utils.checkpoint.checkpoint(self.middle_conv2)
        self.dec_conv7 = torch.utils.checkpoint.checkpoint(self.dec_conv7)
        self.dec_conv8 = torch.utils.checkpoint.checkpoint(self.dec_conv8)
        self.dec_conv9 = torch.utils.checkpoint.checkpoint(self.dec_conv9)
        self.dec_conv10 = torch.utils.checkpoint.checkpoint(self.dec_conv10)
        self.dec_conv11 = torch.utils.checkpoint.checkpoint(self.dec_conv11)
        self.dec_conv12 = torch.utils.checkpoint.checkpoint(self.dec_conv12)
        self.dec_conv13 = torch.utils.checkpoint.checkpoint(self.dec_conv13)
        self.dec_conv14 = torch.utils.checkpoint.checkpoint(self.dec_conv14)
