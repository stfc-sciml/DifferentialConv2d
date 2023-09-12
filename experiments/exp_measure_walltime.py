import torch
from time import time

from exp_utils import UNet

conv2d_methods = {
    'Zero': {'class': 'pad', 'padding_mode': 'zeros'},
    'Refl': {'class': 'pad', 'padding_mode': 'reflect'},
    'Repl': {'class': 'pad', 'padding_mode': 'replicate'},
    'Circ': {'class': 'pad', 'padding_mode': 'circular'},
    'Extr': {'class': 'extra'},
    'Rand': {'class': 'rand'},
    'Part': {'class': 'partial'},
    'EBH': {'class': 'explicit'},
    'Diff': {'class': 'diff',
              'keep_img_grad_at_invalid': False,
              'train_edge_kernel': False,
              'optimized_for': 'speed'}
}


def train(method, device='cuda', image_size=224, batch_size=64, n_batches=20):
    """ train with dummy data """
    x = torch.rand((batch_size, 3, image_size, image_size)).to(device)
    y = torch.rand((batch_size, 3, image_size, image_size)).to(device)

    model = UNet(in_channels=3, out_channels=3,
                 conv2d_method=conv2d_methods[method],
                 seed=0, bias=True,
                 activation=torch.nn.Tanh).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_func = torch.nn.MSELoss()
    _ = model.forward(x)

    t0 = time()
    for _ in range(n_batches):
        y_pred = model.forward(x)
        loss = loss_func(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t = time() - t0
    return t


if __name__ == '__main__':
    wt_ref = None
    for m in conv2d_methods.keys():
        wt = train(m)
        if m == 'Zero':
            wt_ref = wt
        print(m, wt / wt_ref)
