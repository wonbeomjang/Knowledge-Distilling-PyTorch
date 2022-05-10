import os
import argparse
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from model import Model, Regressor

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--params_dir', type=str, default="params", help='the directory of hyper parameters')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                    help="path to saved models (to continue training)")
parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')
parser.add_argument('--distill', nargs='?', const=True, default=False, help='load and train quantizable model')
parser.add_argument('--model_name', type=str, default='vgg11', help='model_name')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
config = parser.parse_args()

checkpoint_dir = os.path.join(config.checkpoint_dir)
if os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def regressor_criterion(preds, targets, guide, hint):
    kd_loss = TF.mse_loss(guide, hint.detach())
    class_loss = TF.cross_entropy(preds, targets)
    return class_loss + kd_loss


def fitnet_criterion(preds, targets, guide, hint):
    kd_loss = TF.mse_loss(guide, hint.detach())
    return kd_loss


def train(net: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
          device: torch.device):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    best = 0

    for epoch in range(config.epoch):
        acc_meter.reset()
        loss_meter.reset()
        pbar = tqdm(data_loader, total=len(data_loader))

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            preds = net(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = preds.argmax(dim=1)
            accuracy = (preds == targets).sum() / len(targets)

            acc_meter.update(accuracy.item())
            loss_meter.update(loss.item())
            pbar.set_description(f"[{epoch}/{config.epoch}] Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.4f}")

        state_dict = {"state_dict": net.state_dict(), "model": net}
        if best < acc_meter.avg:
            best = acc_meter.avg
            torch.save(state_dict, os.path.join(checkpoint_dir, 'teacher_best.pth'))
        torch.save(state_dict, os.path.join(checkpoint_dir, 'teacher_last.pth'))


def distillation(student: Model, teacher: Model, regressor: Regressor, data_loader: DataLoader, criterion: Callable,
                 optimizer: torch.optim.Optimizer, device: torch.device):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    best = 0

    for epoch in range(config.epoch):
        acc_meter.reset()
        loss_meter.reset()
        pbar = tqdm(data_loader, total=len(data_loader))

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            hint = teacher.feature(images)
            guide = regressor(student.feature(images))
            preds = teacher(images)

            loss = criterion(preds, targets, guide, hint)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = preds.argmax(dim=1)
            accuracy = (preds == targets).sum() / len(targets)

            acc_meter.update(accuracy.item())
            loss_meter.update(loss.item())
            pbar.set_description(f"[{epoch}/{config.epoch}] Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.4f}")

        if best < acc_meter.avg:
            best = acc_meter.avg
            torch.save(state_dict, os.path.join(checkpoint_dir, 'student_best.pth'))
        torch.save(state_dict, os.path.join(checkpoint_dir, 'student_best.pth'))


def test(net: nn.Module, data_loader: DataLoader, device: torch.device):
    acc_meter = AverageMeter()
    pbar = tqdm(data_loader, total=len(data_loader))

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        preds = net(images)

        preds = preds.argmax(dim=1)
        accuracy = (preds == targets).sum() / len(targets)

        acc_meter.update(accuracy.item())
        pbar.set_description(f"Validate... Acc: {acc_meter.avg:.4f}")


class ToRGB:
    def __call__(self, pic):
        return pic.repeat(3, 1, 1)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        ToRGB()
    ])

    net = Model(config.model_name)
    data_loader = DataLoader(datasets.MNIST('./', transform=transform, download=True), batch_size=config.batch_size)
    test_loader = DataLoader(datasets.MNIST('./', transform=transform, train=False, download=True),
                             batch_size=config.batch_size)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if not config.distill:
        optimizer = torch.optim.Adam(net.parameters())
        train(net, data_loader, nn.CrossEntropyLoss(), optimizer, device)
    else:
        state_dict = torch.load(os.path.join(checkpoint_dir, 'teacher_best.pth'))
        teacher: Model = state_dict["model"]
        teacher.load_state_dict(state_dict["state_dict"])

        for param in teacher.parameters():
            param.requires_grad = False

        image = torch.zeros((1, 3, config.image_size, config.image_size))
        teacher_output = teacher(image)
        student_output = net(image)
        regressor = Regressor(student_output.size(1), teacher_output.size(1))

        for param in net.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(regressor.parameters())
        print('[*] Train Regressor')
        distillation(net, teacher, regressor, data_loader, regressor_criterion, optimizer, device)

        for param in net.parameters():
            param.requires_grad = True

        for param in regressor.parameters():
            param.requires_grad = False

        print('[*] Train Student')
        optimizer = torch.optim.Adam(net.parameters())
        distillation(net, teacher, regressor, data_loader, fitnet_criterion, optimizer, device)
    test(net, test_loader, device)