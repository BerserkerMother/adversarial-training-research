# transformer multi head attention visualization
# BerserkerMother

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda import amp

import os

from models.transformer import TransformerEncoder
from optimizer.optimizer import Linear_Warmup_Wrapper, ScheduledOptim, Cosine_Warmup_Wrapper
from data.transform import ToTiles

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    T = [
        transforms.Resize(36),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ToTiles(image_size=36, num_tile=9)
    ]

    data = datasets.CIFAR10(root='./cifar-10', train=True, download=True, transform=transforms.Compose(T))
    data_test = datasets.CIFAR10(root='./cifar-10', train=False, download=True, transform=transforms.Compose(T))

    data_loader = DataLoader(
        data,
        batch_size=512,
        shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        data_test,
        batch_size=512,
        shuffle=False,
        pin_memory=True
    )

    network = TransformerEncoder().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    scheduler = Cosine_Warmup_Wrapper(optimizer=optimizer, lr=1e-4)
    scaler = amp.GradScaler()

    num_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    num_parameters /= 1000000

    print("Number of parameters %d M" % num_parameters)
    num_epochs = 100
    print_freq = 10

    e = 0
    if os.path.exists('./checkpoint.pth'):
        checkpoint = torch.load('./checkpoint.pth')
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        e = checkpoint['epoch']

    for i in range(e, num_epochs):
        network.train()
        total_loss = 0.
        for k, (images, targets) in enumerate(data_loader):
            B = data_loader.batch_size
            images, targets = images.to(device), targets.to(device)

            with amp.autocast():
                outputs, _ = network(images)
                loss = F.cross_entropy(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if k % print_freq == 0 and k != 0:
                print("Epoch %2d [%4d/%4d] | Loss: %.4f | lr: %.6f"
                      % (i, k, len(data) / B, total_loss, scheduler.get_current_lr()[0]))
                total_loss = 0

        accuracy_train = evaluate(network, data_loader)
        test_accuracy = evaluate(network, test_loader)
        checkpoint = {
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': i,
            'acc': test_accuracy
        }

        torch.save(checkpoint, './checkpoint.pth')

        print('Epoch %d, train accuracy: %.2f%% | test accuracy: %.2f%%'
              % (i, accuracy_train * 100, test_accuracy * 100))


def evaluate(network, loader):
    network.eval()
    correct = 0.
    num = 0
    with torch.no_grad():
        for images, targets in loader:
            B = images.size(0)
            images, targets = images.to(device), targets.to(device)

            outputs, _ = network(images)
            correct += (outputs.max(1)[1] == targets).sum()
            num += B

    return correct / num


main()
