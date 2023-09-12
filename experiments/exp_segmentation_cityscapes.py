import argparse
from collections import namedtuple
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp_utils import UNet, conv2d_methods

###########################################################################
# Copied from
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
###########################################################################

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True,
          (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

###########################################################################
# Copied from
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
###########################################################################


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
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='epochs')
    parser.add_argument('-l', '--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help='batch size')
    parser.add_argument('-D', '--downscale', type=int, default=2,
                        help='step to downscale image')
    args = parser.parse_args()

    # for a map from color to label
    all_colors = torch.tensor([label.color for label in labels],
                              dtype=torch.float, device=args.device) / 255.
    all_labels = torch.tensor([label.categoryId for label in labels],
                              dtype=torch.long, device=args.device)
    n_classes = 8


    def color_2_label(color):
        b, c, h, w = color.shape
        color = color.moveaxis(1, 3).reshape(-1, 3)
        dist = (color[:, None, :] - all_colors[None, :, :]).norm(dim=2)
        index = dist.argmin(dim=1)
        label = all_labels[index].reshape(b, h, w)
        return label


    # dataset
    d = args.downscale
    image_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
        transforms.Lambda(lambda img: img[:, ::d, ::d])
    ])
    label_trans = transforms.Compose([
        transforms.ToTensor(),
        # labels are rgba
        transforms.Lambda(lambda img: img[:3, ::d, ::d])
    ])
    train_set = torchvision.datasets.Cityscapes(
        './exp_data/cityscapes', split='train', mode='fine',
        target_type='color', transform=image_trans,
        target_transform=label_trans)
    val_set = torchvision.datasets.Cityscapes(
        './exp_data/cityscapes', split='val', mode='fine',
        target_type='color', transform=image_trans,
        target_transform=label_trans)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False)

    # model
    model = UNet(in_channels=3, out_channels=n_classes, u_depth=4,
                 conv2d_method=conv2d_methods[args.conv2d_method],
                 seed=args.seed, activation=torch.nn.Tanh).to(args.device)

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # results
    res_dir = Path('./exp_results/cityscapes')
    res_dir.mkdir(exist_ok=True)
    step_losses = []
    epoch_losses = []

    # training
    model.train()
    torch.manual_seed(42)  # control shuffle
    for epoch in range(args.epochs):
        epoch_loss = 0.
        pbar_batch = tqdm(train_loader, desc=f'EPOCH {epoch + 1}')
        for X, Y in pbar_batch:
            # loss
            X, Y = X.to(args.device), Y.to(args.device)
            Y = color_2_label(Y)
            Y_pred = model.forward(X)
            loss = loss_func(Y_pred, Y)
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            epoch_loss += loss.item()
            step_losses.append(loss.item())
            pbar_batch.set_postfix_str(f'loss={loss.item():.4f}')
        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)

    # save model
    torch.save(
        model.state_dict(),
        res_dir / f'{args.conv2d_method}_seed{args.seed}.weights.pt')

    # evaluate model
    model.eval()
    val_labels_true = []
    val_labels_pred = []
    for X, Y in tqdm(val_loader, desc='EVAL'):
        X, Y = X.to(args.device), Y.to(args.device)
        Y = color_2_label(Y)
        val_labels_true.append(Y)
        with torch.no_grad():
            Y_pred = model(X)
        Y_pred = torch.argmax(Y_pred, dim=1)
        val_labels_pred.append(Y_pred)
    val_labels_true = torch.cat(val_labels_true, dim=0).cpu()
    val_labels_pred = torch.cat(val_labels_pred, dim=0).cpu()

    # segmentation metrics
    tp, fp, fn, tn = smp.metrics.get_stats(val_labels_pred, val_labels_true,
                                           mode='multiclass',
                                           num_classes=n_classes)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2,
                                       reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    # save history and metrics
    torch.save(
        {'step_losses': step_losses,
         'epoch_losses': epoch_losses,
         'iou_score': iou_score,
         'f1_score': f1_score,
         'f2_score': f2_score,
         'accuracy': accuracy,
         'recall': recall},
        res_dir / f'{args.conv2d_method}_seed{args.seed}.hist.pt',
    )
