import os
import torch
import torch.nn as nn
from torchvision.models import vgg


class Regressor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Regressor, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        return self.layer(x)


class Model(nn.Module):
    def __init__(self, backbone="vgg11", num_classes=10):
        super(Model, self).__init__()
        if backbone == "vgg11":
            self.net = vgg.vgg11(pretrained=True)
        elif backbone == "vgg13":
            self.net = vgg.vgg13(pretrained=True)
        elif backbone == "vggb16":
            self.net = vgg.vgg16(pretrained=True)
        elif backbone == "vggb19":
            self.net = vgg.vgg19(pretrained=True)
        else:
            raise Exception("Unsupported Model")

        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def feature(self, x):
        return self.net.features(x)

    def load_params(self, path):
        if not os.path.exists(path):
            print(f"[*] There is no params in {path}")
            return
        self.load_state_dict(torch.load(path, map_location=self.device))