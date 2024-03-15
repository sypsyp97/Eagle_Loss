import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    UNet architecture for image reconstruction.

    Attributes:
        encoder1-4: Sequential layers for the encoding path.
        bottom: The bottleneck layer of the network.
        up1-4: Sequential layers for the decoding path.
        final: Convolutional layer to produce the final output.
        maxpool: MaxPooling layer for downsampling in the encoding path.
        upsample: Upsample layer for upsampling in the decoding path.

    The input to this model should be a tensor of shape (N, 1, H, W), where
    N is the batch size, 1 is the number of channels, and H, W are the height
    and width of the image, respectively.

    Example:
        model = UNet()
        output = model(input_tensor)

    Note:
        This implementation reduces the number of channels at each layer to manage
        model size and complexity, making it potentially more efficient for certain
        reconstruction tasks.
    """

    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = CBR(1, 32)
        self.encoder2 = CBR(32, 64)
        self.encoder3 = CBR(64, 128)
        self.encoder4 = CBR(128, 256)

        self.bottom = CBR(256, 512)

        self.up4 = CBR(512 + 256, 256)
        self.up3 = CBR(256 + 128, 128)
        self.up2 = CBR(128 + 64, 64)
        self.up1 = CBR(64 + 32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # Bottleneck
        bottom = self.bottom(self.maxpool(enc4))

        # Decoding path
        dec4 = self.up4(torch.cat([self.upsample(bottom), enc4], dim=1))
        dec3 = self.up3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.up2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.up1(torch.cat([self.upsample(dec2), enc1], dim=1))

        return self.final(dec1)
