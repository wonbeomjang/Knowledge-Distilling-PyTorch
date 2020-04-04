import os
import torch
import torch.nn as nn
from torchvision.models import resnet, densenet
from tqdm import tqdm

from utils import AverageMeter


class Model(nn.Module):
    def __init__(self, num_classes, param):
        super(Model, self).__init__()
        if param.model_name == "resnet18": self.net = resnet.resnet18(pretrained=True)
        elif param.model_name == "resnet34": self.net = resnet.resnet34(pretrained=True)
        elif param.model_name == "resnet50": self.net = resnet.resnet50(pretrained=True)
        elif param.model_name == "resnet101": self.net = resnet.resnet101(pretrained=True)
        elif param.model_name == "resnet152": self.net = resnet.resnet152(pretrained=True)
        in_feature = self.net.fc.in_features
        self.net.fc = nn.Linear(in_feature, num_classes)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.param = param

    def forward(self, x):
        return self.net(x)

    def train_model(self, data_loader, criterion, optimizer, teacher_preds, param):
        self.train()
        self.to(self.device)
        loss_avg = AverageMeter()
        acc_avg = AverageMeter()
        loss_avg.reset()
        acc_avg.reset()

        for step, (images, targets) in tqdm(enumerate(data_loader)):
            images: torch.Tensor = images.to(self.device)
            targets: torch.Tensor = targets.to(self.device)

            preds: torch.Tensor = self.forward(images)
            loss: torch.Tensor = criterion(preds, targets, teacher_preds[step], param)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.mean().item())
            acc_avg.update((preds == targets).mean().item())

        return loss_avg.avg, acc_avg.avg

    def validate_model(self, data_loader, criterion, teacher_preds, param):
        with torch.no_grad():
            self.eval()
            self.to(self.device)
            loss_avg = AverageMeter()
            acc_avg = AverageMeter()
            loss_avg.reset()
            acc_avg.reset()

            for step, (images, targets) in tqdm(enumerate(data_loader)):
                images: torch.Tensor = images.to(self.device)
                targets: torch.Tensor = targets.to(self.device)

                preds: torch.Tensor = self.forward(images)
                loss: torch.Tensor = criterion(preds, targets, teacher_preds[step], param)

                loss_avg.update(loss.mean().item())
                acc_avg.update((preds == targets).mean().item())

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
        results = {}

        with torch.no_grad():
            for step, (images, targets) in tqdm(enumerate(data_loader)):
                results[step] = self.forward(images)

        return results

    def load_params(self, path):
        if not os.path.exists(path):
            print(f"[*] There is no params in {path}")
            return
        self.load_state_dict(torch.load(path, map_location=self.device))