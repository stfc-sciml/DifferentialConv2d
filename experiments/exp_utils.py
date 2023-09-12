import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_conv2d.layers import DiffConv2dLayer, PartConv2dLayer, \
    ExplicitConv2dLayer, ExtraConv2dLayer, RandConv2dLayer

# allowed Conv2d types
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
             'keep_img_grad_at_invalid': True,
             'train_edge_kernel': False,
             'optimized_for': 'speed'},
}


def get_conv2d_layer(n_in, n_out, k_size, method_args, seed_layer, bias):
    """ conv2d type handler"""
    # reset seed for each layer to have the same weights in different models
    torch.manual_seed(seed_layer)
    if method_args['class'] == 'pad':
        return nn.Conv2d(n_in, n_out, k_size, padding='same',
                         padding_mode=method_args['padding_mode'], bias=bias)
    elif method_args['class'] == 'extra':
        return ExtraConv2dLayer(n_in, n_out, k_size, bias=bias)
    elif method_args['class'] == 'rand':
        return RandConv2dLayer(n_in, n_out, k_size, bias=bias)
    elif method_args['class'] == 'partial':
        return PartConv2dLayer(n_in, n_out, k_size, bias=bias)
    elif method_args['class'] == 'explicit':
        return ExplicitConv2dLayer(n_in, n_out, k_size, bias=bias)
    elif method_args['class'] == 'diff':
        return DiffConv2dLayer(
            n_in, n_out, k_size,
            keep_img_grad_at_invalid=method_args['keep_img_grad_at_invalid'],
            train_edge_kernel=method_args['train_edge_kernel'], bias=bias,
            optimized_for=method_args['optimized_for'])
    else:
        assert False, f'Unknown conv2d class: {method_args["class"]}.'


def get_conv2d_block(n_in, n_out, k_size, method_args, seed_block, bias,
                     separate_act_bn=False, activation=None):
    """ block in UNet """
    if activation is None:
        activation = nn.ReLU
    if not separate_act_bn:
        return nn.Sequential(
            get_conv2d_layer(n_in, n_out, k_size, method_args, seed_block,
                             bias),
            activation(),
            nn.BatchNorm2d(num_features=n_out),
            get_conv2d_layer(n_out, n_out, k_size, method_args, seed_block + 1,
                             bias),
            activation(),
            nn.BatchNorm2d(num_features=n_out))
    else:
        return nn.Sequential(
            get_conv2d_layer(n_in, n_out, k_size, method_args, seed_block,
                             bias),
            activation(),
            nn.BatchNorm2d(num_features=n_out),
            get_conv2d_layer(n_out, n_out, k_size, method_args, seed_block + 1,
                             bias),
        ), activation(), nn.BatchNorm2d(num_features=n_out)


class UNet(nn.Module):
    """ class UNet """

    def __init__(self, in_channels, out_channels, kernel_size=3, u_depth=4,
                 out_channels_first=64, conv2d_method=None, bias=True,
                 activation=None, seed=0):
        """
        Constructor
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param u_depth: depth of u-net
        :param out_channels_first: output channels of first block
        :param conv2d_method: args for conv2d layers
        :param bias: consider bias in conv2d layers
        :param activation: activation function
        :param seed: random seed for model weight initialization
        """
        super(UNet, self).__init__()
        if conv2d_method is None:
            conv2d_method = {'class': 'pad', 'padding_mode': 'zeros'}

        # first layer arguments
        in_chs, out_chs = in_channels, out_channels_first
        seed_block = seed * 1000000

        # encoder
        self.enc_conv2d_list = nn.ModuleList()
        for _ in range(u_depth):
            self.enc_conv2d_list.append(get_conv2d_block(
                in_chs, out_chs, kernel_size, conv2d_method, seed_block,
                bias, activation=activation))
            in_chs = out_chs
            out_chs = out_chs * 2
            seed_block += 2

        # middle
        self.middle_conv, self.middle_act, self.middle_bn = \
            get_conv2d_block(in_chs, out_chs, kernel_size,
                             conv2d_method, seed_block, bias,
                             separate_act_bn=True, activation=activation)
        seed_block += 2

        # decoder
        self.dec_upsample_list = nn.ModuleList()
        self.dec_conv1x1_list = nn.ModuleList()
        self.dec_conv2d_list = nn.ModuleList()
        for _ in range(u_depth):
            self.dec_upsample_list.append(
                nn.Upsample(scale_factor=2, mode='bilinear'))
            self.dec_conv1x1_list.append(
                nn.Conv2d(out_chs, in_chs, kernel_size=1, bias=bias))
            self.dec_conv2d_list.append(get_conv2d_block(
                out_chs, in_chs, kernel_size, conv2d_method, seed_block, bias,
                activation=activation))
            out_chs = in_chs
            in_chs = in_chs // 2
            seed_block += 2

        # last conv
        self.last = nn.Conv2d(out_chs, out_channels, kernel_size=1, bias=bias)

    def forward(self, x, returns_middle=False):
        """ forward """
        # encoder
        x_enc_list = []
        for conv2d in self.enc_conv2d_list:
            x = conv2d(x)
            x_enc_list.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        # middle
        mid = self.middle_act(self.middle_conv(x))
        x = self.middle_bn(mid)

        # decoder
        for upsample, conv1x1, conv2d, x_enc in zip(
                self.dec_upsample_list, self.dec_conv1x1_list,
                self.dec_conv2d_list, x_enc_list[::-1]):
            # transpose conv
            x = conv1x1(upsample(x))
            # cat with encoder output
            x = torch.cat([x, x_enc], dim=1)
            # conv2d
            x = conv2d(x)

        # last
        last = self.last(x)
        if returns_middle:
            return mid, last
        else:
            return last


if __name__ == "__main__":
    img = torch.rand(2, 1, 64, 64)
    model = UNet(in_channels=img.shape[1], out_channels=10, u_depth=4,
                 conv2d_method={'class': 'diff',
                                'keep_img_grad_at_invalid': True,
                                'train_edge_kernel': False,
                                'optimized_for': 'speed'})
    mid_, last_ = model.forward(img, returns_middle=True)
    print(mid_.shape)
    print(last_.shape)
