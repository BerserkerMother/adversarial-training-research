# transformer multi head attention visualization
# BerserkerMother

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.transformer import TransformerEncoder
from optimizer.optimizer import Linear_Warmup_Wrapper, ScheduledOptim, Cosine_Warmup_Wrapper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    T = [
        transforms.Resize(36),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
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
    optimizer = Cosine_Warmup_Wrapper(optimizer=optimizer, d_model=network.hidden_size)

    num_parameters = 0
    for param in network.parameters():
        size = param.view(-1).size()[0]
        num_parameters += size

    print("Number of parameters %d" % num_parameters)
    num_epochs = 100
    print_freq = 10

    for i in range(num_epochs):
        network.train()
        total_loss = 0.
        for k, (images, targets) in enumerate(data_loader):
            B = data_loader.batch_size
            images, targets = images.to(device), targets.to(device)

            outputs = network(images)
            loss = F.cross_entropy(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if k % print_freq == 0 and k != 0:
                print("Epoch %2d [%4d/%4d] | Loss: %.4f | lr: %.6f"
                      % (i, k, len(data) / B, total_loss, optimizer.get_current_lr()[0]))
                total_loss = 0

        accuracy_train = evaluate(network, data_loader)
        test_accuracy = evaluate(network, test_loader)
        checkpoint = {
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': i,
            'acc': test_accuracy
        }

        torch.save(checkpoint, './checkpoint%d.pth' % i)

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

            outputs = network(images)
            correct += (outputs.max(1)[1] == targets).sum()
            num += B

    return correct / num


main()
