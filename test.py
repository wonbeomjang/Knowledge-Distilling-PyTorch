import argparse
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from models import Model
from loss import loss_kd
from dataloader import get_test_loader
from utils import Params, AverageMeter
from tqdm import tqdm

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--params_dir', type=str, default="params", help='the directory of hyper parameters')
parser.add_argument('-m', '--model_name', type=str, default='resnet101', help='the name of backbone network')
parser.add_argument('--log_path', type=str, default='logs', help="directory to save train log")
parser.add_argument('--epoch', type=int, default=0, help='value of current epoch')
parser.add_argument('--num_epoch', type=int, default=89, help='the number of epoch in train')
parser.add_argument('--decay_epoch', type=int, default=30, help='the number of decay epoch in train')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--num_classes', type=int, default=10, help='the number of classes')
parser.add_argument('--is_distill', type=bool, default=True)
args = parser.parse_args()

if __name__ == '__main__':
    params = Params(os.path.join(args.params_dir, f'{args.student_name}.json'))
    acc = AverageMeter()
    net = Model(args.num_classes, params)
    net.load_params(os.path.join(args.checkpoint_dir, params.model_name, f'final.pth'))

    writer = SummaryWriter(args.log_path)
    criterion = loss_kd
    test_loader = get_test_loader(args.image_size, 1024)
    num_correct = 0
    num_data = 0

    for images, targets in tqdm(test_loader, desc=f'{params.model_name} Testing...'):
        images: torch.Tensor = images.to(net.device)
        targets: torch.Tensor = targets.to(net.device)
        preds: torch.Tensor = net.predict_image(images)

        acc.update((preds == targets).sum().item()/images.shape[0])

    print(f'{acc.avg * 100}%')
