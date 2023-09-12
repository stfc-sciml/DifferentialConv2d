import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from exp_utils import UNet, conv2d_methods


class UNetClassify(nn.Module):
    """ UNet + classifier in the middle """

    def __init__(self, conv2d_method, seed, n_classes, image_size, u_depth=4,
                 out_channels_first=64):
        super(UNetClassify, self).__init__()
        self.unet = UNet(in_channels=3, out_channels=3, kernel_size=3,
                         u_depth=u_depth,
                         out_channels_first=out_channels_first,
                         conv2d_method=conv2d_method, seed=seed)
        factor = 2 ** u_depth
        n_mid = out_channels_first * factor * (image_size // factor) ** 2
        self.fc1 = nn.Linear(n_mid, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        mid, last = self.unet.forward(x, returns_middle=True)
        mid = mid.reshape(len(mid), -1)
        mid = F.relu(self.fc1(mid))
        mid = self.fc2(mid)
        return mid, last


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('-m', '--conv2d-method', type=str,
                        choices=conv2d_methods.keys(), required=True,
                        help='method for Conv2d.')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='device')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='epochs')
    parser.add_argument('-l', '--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('-S', '--image-size', type=int, default=448,
                        help='image size')
    parser.add_argument('-r', '--beta-reconstruct', type=float, default=0.01,
                        help='beta for reconstruction loss')
    args = parser.parse_args()

    # dataset
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.Caltech101('./exp_data/',
                                              download=True,
                                              transform=data_transforms)

    # split
    generator1 = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(dataset, [.8, .2], generator=generator1)

    # loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False)

    # model
    model = UNetClassify(conv2d_methods[args.conv2d_method],
                         args.seed, n_classes=101,
                         image_size=args.image_size).to(args.device)

    # optimizer and losses
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_recon = torch.nn.MSELoss()

    # results
    res_dir = Path('./exp_results/caltech101')
    res_dir.mkdir(exist_ok=True)
    step_losses = []
    epoch_losses = []

    # training
    model.train()
    torch.manual_seed(42)  # control shuffle
    for epoch in tqdm(range(args.epochs), desc='EPOCHS'):
        epoch_loss = 0.
        pbar_batch = tqdm(train_loader, total=len(train_loader),
                          desc=f'EPOCH {epoch}')
        for X, Y in pbar_batch:
            # loss
            X, Y = X.to(args.device), Y.to(args.device)
            Y_pred, X_pred = model.forward(X)
            loss_cls = loss_class(Y_pred, Y)
            loss_rec = loss_recon(X_pred, X)
            loss = loss_cls + loss_rec * args.beta_reconstruct
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            epoch_loss += loss.item()
            step_losses.append(loss.item())
            pbar_batch.set_postfix_str(
                f'loss_cls={loss_cls.item():.2f}; '
                f'loss_rec={loss_rec.item():.2f}')
        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)

    # save last model
    torch.save(
        model.state_dict(),
        res_dir / f'{args.conv2d_method}_seed{args.seed}.weights.pt')

    # evaluate last model
    model.eval()
    test_Y_true = []
    test_Y_pred = []
    for X, Y in tqdm(test_loader, total=len(test_loader), desc='EVAL'):
        X, Y = X.to(args.device), Y.to(args.device)
        test_Y_true.append(Y)
        with torch.no_grad():
            Y_pred, X_pred = model(X)
        Y_pred = torch.argmax(Y_pred, dim=1)
        test_Y_pred.append(Y_pred)
    test_Y_true = torch.cat(test_Y_true, dim=0)
    test_Y_pred = torch.cat(test_Y_pred, dim=0)
    accuracy = ((test_Y_pred == test_Y_true).sum() / len(test_Y_true)).item()

    # save history
    torch.save(
        {'step_losses': step_losses,
         'epoch_losses': epoch_losses,
         'accuracy': accuracy},
        res_dir / f'{args.conv2d_method}_seed{args.seed}.hist.pt',
    )
