import os
import torch
import torch.nn as nn
from torchvision.models import resnet, densenet, vgg
from tqdm import tqdm, tqdm_notebook

from utils import AverageMeter


def conv_block(in_channels, out_channels, batch_norm=True):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                          nn.ReLU(),
                          nn.BatchNorm2d(out_channels))
    return block


class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32),
                                      nn.MaxPool2d(2),
                                      conv_block(32, 64),
                                      conv_block(64, 64),
                                      nn.MaxPool2d(2),
                                      conv_block(64, 128),
                                      conv_block(128, 128),
                                      nn.AdaptiveAvgPool2d(7))
        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 128, 1024),
                                        nn.Dropout(),
                                        nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes, param, epoch=0):
        super(Model, self).__init__()
        if param.model_name == "resnet18":
            self.net = resnet.resnet18(pretrained=True)
        elif param.model_name == "resnet34":
            self.net = resnet.resnet34(pretrained=True)
        elif param.model_name == "resnet50":
            self.net = resnet.resnet50(pretrained=True)
        elif param.model_name == "resnet101":
            self.net = resnet.resnet101(pretrained=True)
        elif param.model_name == "resnet152":
            self.net = resnet.resnet152(pretrained=True)
        elif param.model_name == "densenet121":
            self.net = densenet.densenet121(pretrained=True)
        elif param.model_name == "vgg11":
            self.net = vgg.vgg11_bn(pretrained=True)
        elif param.model_name == "vggb13":
            self.net = vgg.vgg13_bn(pretrained=True)
        elif param.model_name == "vggb13":
            self.net = vgg.vgg19_bn(pretrained=True)
        else:
            self.net = SimpleModel(num_classes)

        if 'resnet' in param.model_name:
            in_feature = self.net.fc.in_features
            self.net.fc = nn.Linear(in_feature, num_classes)
        if 'densenet' in param.model_name:
            in_feature = self.net.classifier.in_features
            self.net.classifier = nn.Linear(in_feature, num_classes)
        if 'vgg11' in param.model_name:
            self.net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.param = param
        self.epoch = epoch
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

    def train_model(self, data_loader, criterion, optimizer, teacher_preds, param):
        self.train()
        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        loss_avg.reset()
        acc_avg.reset()

        for step, (images, targets) in enumerate(tqdm(data_loader, desc=f'Train Epoch {self.epoch}')):
            images: torch.Tensor = images.to(self.device)
            targets: torch.Tensor = targets.to(self.device)

            preds: torch.Tensor = self.forward(images)
            loss: torch.Tensor = criterion(preds, targets, torch.from_numpy(teacher_preds[step]).type_as(preds), param)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = preds.argmax(dim=1)
            loss_avg.update(loss.mean().item())
            acc_avg.update((preds == targets).sum().item() / images.shape[0])
        self.epoch += 1

        return loss_avg.avg, acc_avg.avg

    def validate_model(self, data_loader, criterion, teacher_preds, param):
        with torch.no_grad():
            self.eval()
            loss_avg = AverageMeter()
            acc_avg = AverageMeter()
            loss_avg.reset()
            acc_avg.reset()

            for step, (images, targets) in enumerate(tqdm(data_loader, desc=f'Validation Epoch {self.epoch}')):
                images: torch.Tensor = images.to(self.device)
                targets: torch.Tensor = targets.to(self.device)

                preds: torch.Tensor = self.forward(images)
                loss: torch.Tensor = criterion(preds, targets, torch.from_numpy(teacher_preds[step]).to(self.device),
                                               param)

                preds = preds.argmax(dim=1)
                loss_avg.update(loss.mean().item())
                acc_avg.update((preds == targets).sum().item() / images.shape[0])

            return loss_avg.avg, acc_avg.avg

    def predict_image(self, image: torch.Tensor):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.eval()
        self.to(device)

        pred: torch.Tensor = self.forward(image)
        pred = pred.argmax(dim=1)

        return pred

    def fetch_output(self, data_loader):
        self.eval()
        results = []

        for images, targets in tqdm(data_loader, desc=f'Fetch answer'):
            images = images.to(self.device)
            results += [self.forward(images).detach().cpu().numpy()]

        return results

    def load_params(self, path):
        if not os.path.exists(path):
            print(f"[*] There is no params in {path}")
            return
        self.load_state_dict(torch.load(path, map_location=self.device))
