from torchvision.transforms import transforms
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


def get_loader(image_size, batch_size, data_set='cifar10'):
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    if data_set == 'cifar100':
        dataset_class = CIFAR100
    elif data_set == 'cifar10':
        dataset_class = CIFAR10
    elif data_set == 'mnist':
        dataset_class = MNIST
    else:
        raise Exception('No matched dataset')

    dataset = dataset_class('./dataset', train=True, transform=transform, download=True)
    train_length = int(0.9 * len(dataset))
    validation_length = len(dataset) - train_length

    train_dataset, validation_dataset = random_split(dataset, (train_length, validation_length))
    train_loader = DataLoader(train_dataset, batch_size, False)
    validation_loader = DataLoader(validation_dataset, batch_size, False)

    return train_loader, validation_loader


def get_test_loader(image_size, batch_size, data_set='cifar10'):
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    if data_set == 'cifar100':
        dataset_class = CIFAR100
    elif data_set == 'cifar10':
        dataset_class = CIFAR10
    elif data_set == 'mnist':
        dataset_class = MNIST
    else:
        raise Exception('No matched dataset')
    return DataLoader(dataset_class('./dataset', train=False, transform=transform, download=True), batch_size, False)
