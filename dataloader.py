from torchvision.transforms import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


def get_loader(image_size, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    dataset = MNIST('./dataset', train=True, transform=transform, download=True)
    train_length = int(0.9 * len(dataset))
    validation_length = len(dataset) - train_length

    train_dataset, validation_dataset = random_split(dataset, (train_length, validation_length))
    train_loader = DataLoader(train_dataset, batch_size, True)
    validation_loader = DataLoader(validation_dataset, batch_size, True)

    return train_loader, validation_loader


def get_test_loader(image_size, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return DataLoader(MNIST('./dataset', train=False, transform=transform, download=True), batch_size, False)
