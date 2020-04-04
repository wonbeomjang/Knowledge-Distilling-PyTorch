import argparse
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from models import Model
from loss import loss_kd
from dataloader import get_loader
from utils import Params

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--params_dir', type=str, default="params", help='the directory of hyper parameters')
parser.add_argument('--student_name', type=str, help='the name of backbone network')
parser.add_argument('--teacher_name', type=str, help='the name of backbone network')
parser.add_argument('--log_path', type=str, default='logs', help="directory to save train log")
parser.add_argument('--epoch', type=int, default=0, help='value of current epoch')
parser.add_argument('--num_epoch', type=int, default=0, help='the number of epoch in train')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--num_classes', type=int, default=10, help='the number of classes')
parser.add_argument('--is_distill', type=bool, default=True)
args = parser.parse_args()

if __name__ == '__main__':
    student_params = Params(os.path.join(args.params_dir, f'{args.student_name}.json'))
    teacher_params = Params(os.path.join(args.params_dir, f'{student_params.teacher_name}.json'))

    student = Model(args.num_classes, student_params)
    student.load_params(os.path.join(args.checkpoint_dir, student_params.model_name, f'{args.epoch}.pth'))

    teacher = Model(args.num_classes, teacher_params)
    teacher.load_params(os.path.join(args.checkpoint_dir, teacher_params.model_name, f'final.pth'))

    summary_title = f'{student_params.teacher_name}_teaches_{student_params.model_name} '

    if not os.path.exists(os.path.join(args.checkpoint_dir, student_params.model_name)):
        os.makedirs(os.path.join(args.checkpoint_dir, student_params.model_name))
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    criterion = loss_kd
    optimizer = torch.optim.Adam(student.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_epoch)
    train_loader, validation_loader = get_loader(args.data_dir, args.image_size, student_params.batch_size)
    scheduler.step(args.epoch)

    teacher_train_ans = teacher.fetch_output(train_loader)
    teacher_val_ans = teacher.fetch_output(validation_loader)

    for iter in range(args.epoch, args.num_epoch):
        print(iter)
        train_loss, train_acc = student.train_model(train_loader, criterion, optimizer, teacher_train_ans, student_params)
        validation_loss, validation_acc = student.validate_model(validation_loader, criterion, teacher_val_ans, student_params)
        writer.add_scalars(f'{summary_title}/Loss', {'train': train_loss, 'val': validation_loss}, iter)
        writer.add_scalars(f'{summary_title}/Accuracy', {'train': train_acc, 'val': validation_acc}, iter)
        torch.save(student.state_dict(), os.path.join(args.checkpoint_dir, student_params.model_name, f'{iter}.pth'))
        scheduler.step()
