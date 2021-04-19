# transformer multi head attention visualization
# BerserkerMother

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda import amp

import os
import argparse

from models.transformer import TransformerEncoder
from optimizer.optimizer import Linear_Warmup_Wrapper, ScheduledOptim, Cosine_Warmup_Wrapper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    train_transform = [
        transforms.RandomResizedCrop((args.image_size, args.image_size)),
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]

    test_transform = [
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]

    data = datasets.CIFAR10(root=args.data, train=True, download=True,
                            transform=transforms.Compose(train_transform))
    data_test = datasets.CIFAR10(root=args.data, train=False, download=True,
                                 transform=transforms.Compose(test_transform))

    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        data_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    network = TransformerEncoder(image_size=args.image_size, hidden_size=args.hidden_size, num_head=args.num_heads,
                                 attention_size=args.attention_size, num_encoder_layers=args.num_encoder_layers,
                                 dropout=args.dropout, num_classes=10, div_term=args.div_term).to(device)
    print(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    scheduler = Cosine_Warmup_Wrapper(optimizer=optimizer, lr=args.lr, total_steps=args.total_steps)
    scaler = amp.GradScaler()

    num_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    num_parameters /= 1000000

    print("Number of parameters %dM" % num_parameters)

    e = 0
    if os.path.exists('./checkpoint.pth'):
        checkpoint = torch.load('./checkpoint.pth')
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        e = checkpoint['epoch'] + 1

    for i in range(e, args.num_epochs):
        accuracy_train = train(args, network=network, data_loader=data_loader, optimizer=optimizer,
                               scaler=scaler, scheduler=scheduler, data=data, e=i)
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


def train(args, network, data_loader, optimizer, scaler, scheduler, data, e):
    network.train()
    total_loss, correct, total_num = 0., 0., 0
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

        total_num += images.size(0)
        correct += (outputs.max(1)[1] == targets).sum()

        if k % args.print_freq == 0 and k != 0:
            print("Epoch %2d [%4d/%4d] | Loss: %3.4f | lr: %.6f"
                  % (e, k, len(data) / B, total_loss, scheduler.get_current_lr()[0]))
            total_loss = 0

    return correct / total_num


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


arg_parser = argparse.ArgumentParser(description='ViT model and Attention visualization')
# data related args
arg_parser.add_argument('--data', default='./cifar-10', type=str, help='path to dataset')
arg_parser.add_argument('--batch_size', default=512, type=int, help='number of samples in a batch')
# optimizer related args
arg_parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
arg_parser.add_argument('--total_steps', default=10000, type=int, help='number of step to decay lr in')
# training related args
arg_parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
arg_parser.add_argument('--print_freq', default=10, type=int, help='frequency of logging')
# model related args
arg_parser.add_argument('--image_size', default=36, type=int, help='resize image to')
arg_parser.add_argument('--hidden_size', default=768, type=int, help='number of hidden nodes')
arg_parser.add_argument('--num_heads', default=12, type=int, help='number of attention head')
arg_parser.add_argument('--attention_size', default=768, type=int, help='hidden size of attention fc')
arg_parser.add_argument('--num_encoder_layers', default=12, type=int, help='number of encoders')
arg_parser.add_argument('--dropout', default=.2, type=float, help='dropout rate')
arg_parser.add_argument('--div_term', default=3, type=int, help='div_term^2 is number of tiles')

arg = arg_parser.parse_args()

main(arg)
