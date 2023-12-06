#import matplotlib.pyplot as plt
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim

import torchvision


class Active_wearer_model(nn.Module):    # TODO: dropout, abs?
    def __init__(self, input_dim, output_dim=2):
        super(Active_wearer_model, self).__init__()
        mic_num, h, w = input_dim  # (6, 257, 4 for [0,0])
        # print(f"$$${input_dim=}")
        self.fc1 = nn.Linear(mic_num*h*w, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.fc3 = nn.Linear(64, output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = x.view(x.size(0), -1)     # Flattening the input tensor
        x = x.to(self.fc1.weight.dtype)  # Convert input tensor to the same data type as fc1 weight
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Speaker_segmentation_model(nn.Module):
    def __init__(self, input_dim, output_dim):  # TODO: change to segmentation_model
        super(Speaker_segmentation_model, self).__init__()
        h, w = input_dim  # (360, 640)
        #print(f"$$${input_dim=}")
        self.fc1 = nn.Linear(h*w, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim[0]*output_dim[1]*output_dim[2])

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):   # TODO: change to segmentation_model as well
        x = x.view(x.size(0), -1)     # Flattening the input tensor
        x = x.to(self.fc1.weight.dtype)  # Convert input tensor to the same data type as fc1 weight
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Test_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, output_dim)
        self.dropout1 = nn.Dropout(p=0.3)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        # x = [batch size,sr]
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        return x

    # def eval(self):
    #     self.eval()


class AV_segmentation_model(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AV_segmentation_model, self).__init__()
        self.input_dim = input_dim    # (2, )
        self.output_dim = output_dim

    def forward(self, x):
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=2,
            features=[64, 128, 256, 512],
            av_combination_level="before_unet"
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.av_combination_level = av_combination_level

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        bottleneck_feature = features[-1]
        if self.av_combination_level == "in_unet_bottleneck":
            bottleneck_feature += 1
        self.bottleneck = DoubleConv(bottleneck_feature, features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def concat_av(self, video, audio):
        audio = audio.unsqueeze(1)
        # Concatenate the tensors along the second dimension
        concatenated_av = torch.cat((audio, video), dim=1)
        return concatenated_av


    def forward(self, x, y=None):
        # print(f"x shape before down: {x.shape}")
        # print(f"y shape before down: {y.shape}")
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # print(f"x shape after down: {x.shape}")
        if y is not None:
            x = self.concat_av(x, y)

        # print(f"The shape of x after concat_av: {x.shape}")
        x = self.bottleneck(x)
        # print(f"The shape of x after bottleneck is: {x.shape}")
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class UNET_lightning(pl.LightningModule):
    def __init__(
            self,
            in_channels=3,
            out_channels=2,
            features=[64, 128, 256, 512],
            av_combination_level="before_unet"
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.av_combination_level = av_combination_level

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        bottleneck_feature = features[-1]
        if self.av_combination_level == "in_unet_bottleneck":
            bottleneck_feature += 1
        self.bottleneck = DoubleConv(bottleneck_feature, features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, y=None):

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        if y is not None:
            x = self.concat_av(x, y)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


    def training_step(self, batch, batch_idx):
        pass

def unet_test():
    x = torch.randn((3, 1, 161, 161))
    y = torch.randn((3, 2, 161, 161))
    model = UNET(in_channels=1,
                 out_channels=2,
                 features=[16, 32, 64, 128],       #[64, 128, 256, 512],
                 av_combination_level="before_unet"
                 )
    preds = model(x)
    # print(f"{preds.shape=}")
    # print(f"{x.shape=}")
    assert preds.shape == y.shape


if __name__ == "__main__":
    unet_test()



