from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def cifar10_dataloaders(batch_size=128, num_workers=8):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root='./data', download=True, train=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR10(root='./data', download=True, train=False, transform=test_transform)

    test_loader = DataLoader(test_set, batch_size=int(batch_size/2), shuffle=False, num_workers=int(num_workers/2))

    return train_loader, test_loader
