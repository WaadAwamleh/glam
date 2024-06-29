import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-03, momentum=0.01), # to be similar to Tensorflow that has 0.99 (1-0.01)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-03, momentum=0.01), # to be similar to Tensorflow https://github.com/pytorch/examples/issues/289
            nn.ReLU(inplace=True))


def down():
    """Downscaling with maxpool"""
    return nn.MaxPool2d(2)


def up(in_channels, out_channels):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))

class UNet(nn.Module):
    def __init__(self, attention=True):
        super(UNet, self).__init__()
        self.attention = attention
        n_channels = 1

        # those names must correcpond to the names given to the different layers in the tensorflow model.
        self.conv_1 = double_conv(n_channels, 8)
        self.conv_2 = double_conv(8, 16)
        self.conv_3 = double_conv(16, 32)
        self.conv_4 = double_conv(32, 64)
        self.conv_5 = double_conv(64, 128) # conv_5 has 128 channels after 2 conv
        self.down = down()

        # up-scaling with deconv module
        self.decon_6 = up(128, 64)
        self.decon_7 = up(64, 32)
        self.decon_8 = up(32, 16)
        self.decon_9 = up(16, 8)

        # double conv
        self.conv_6 = double_conv(128, 64)
        self.conv_7 = double_conv(64, 32)
        self.conv_8 = double_conv(32, 16)
        self.conv_9 = double_conv(16, 8)
        
        if self.attention:
            self.attention_gate6 = AttentionGate(F_g=64, F_l=64, F_int=32)
            self.attention_gate7 = AttentionGate(F_g=32, F_l=32, F_int=16)
            self.attention_gate8 = AttentionGate(F_g=16, F_l=16, F_int=8)
            self.attention_gate9 = AttentionGate(F_g=8, F_l=8, F_int=4)
            

        self.final = nn.Conv2d(8, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data) # same initialisation than version using tensorflow
                if m.bias is not None:
                    m.bias.data.zero_()

    def concat(self, x1, x2):
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        upscale1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([upscale1, x2], dim=1)
        return x

    def forward(self, x):
        # Contracting Path
        x1 = self.conv_1(x)
        x = self.down(x1)
        x2 = self.conv_2(x)
        x = self.down(x2)
        x3 = self.conv_3(x)
        x = self.down(x3)
        x4 = self.conv_4(x)
        x = self.down(x4)
        x5 = self.conv_5(x)

        # Expansive Path with optional Attention
        x = self.decon_6(x5)
        if self.attention:
            x = self.concat(self.attention_gate6(x, x4), x)
        else:
            x = self.concat(x4, x)
        x = self.conv_6(x)

        x = self.decon_7(x)
        if self.attention:
            x = self.concat(self.attention_gate7(x, x3), x)
        else:
            x = self.concat(x3, x)
        x = self.conv_7(x)

        x = self.decon_8(x)
        if self.attention:
            x = self.concat(self.attention_gate8(x, x2), x)
        else:
            x = self.concat(x2, x)
        x = self.conv_8(x)

        x = self.decon_9(x)
        if self.attention:
            x = self.concat(self.attention_gate9(x, x1), x)
        else:
            x = self.concat(x1, x)
        x = self.conv_9(x)

        # Final activation
        output = self.final(x)
        output_sigmoid = self.sigmoid(output)
        return output_sigmoid

