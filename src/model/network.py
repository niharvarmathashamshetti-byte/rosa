import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.residual_conv = nn.Identity()
            
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual_conv(x)
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.relu2(out + res)
        return out

class KneeUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        channels: tuple = (16, 32, 64, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        
        self.enc1 = ResidualBlock3D(in_channels, channels[0], dropout=0.0)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = ResidualBlock3D(channels[0], channels[1], dropout=dropout)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = ResidualBlock3D(channels[1], channels[2], dropout=dropout)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = ResidualBlock3D(channels[2], channels[3], dropout=dropout)
        
        self.up3 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(channels[2] + channels[2], channels[2], dropout=dropout)
        
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(channels[1] + channels[1], channels[1], dropout=dropout)
        
        self.up1 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(channels[0] + channels[0], channels[0], dropout=0.0)
        
        self.out_conv = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        
        e3 = self.enc3(d2)
        d3 = self.down3(e3)
        
        b = self.bottleneck(d3)
        
        u3 = self.up3(b)
        u3 = torch.cat([u3, e3], dim=1)
        d_out3 = self.dec3(u3)
        
        u2 = self.up2(d_out3)
        u2 = torch.cat([u2, e2], dim=1)
        d_out2 = self.dec2(u2)
        
        u1 = self.up1(d_out2)
        u1 = torch.cat([u1, e1], dim=1)
        d_out1 = self.dec1(u1)
        
        logits = self.out_conv(d_out1)
        return logits
