# DiffConv2d

This is the code repository for the following paper: 
```
K. Leng and J. Thiyagalingam, Padding-free Convolution based on
Preservation of Differential Characteristics of Kernels.
```

## Installation

**`PyTorch` is the only dependency for applications**. If you need a
`TensorFlow` version, please contact us.

## Usage

**All source files needed for applications are included in `diff_conv2d/`**.

### Part I. Image Filtering by Convolution

If you are using the following `PyTorch` API for image convolution:

```python
output = torch.nn.functional.conv2d(input, weight, padding='same', bias=None,
                                    stride=1, dilation=1, groups=any_groups) 
```

you can then switch to our method by

```python
from diff_conv2d.functional import DiffConv2d

diff = DiffConv2d(kernel_size=3)  # kernel_size can be 3, 5, or 7
output = diff.conv2d(input, weight, groups=any_groups,
                     keep_img_grad_at_invalid=True,
                     edge_kernel=None, optimized_for='memory')
```

The additional arguments are explained below:

* `keep_img_grad_at_invalid`: specifies whether to keep the gradient of
  input
  pixels repeatedly used for boundary handling. For example, when using
  padding by
  `padding_mode=replicate` with a normal `Conv2d` layer, one may consider
  turning
  off the gradient with respect to the replicated pixel values;
  however, `PyTorch` does not offer such an option.
  With our `DiffConv2d.conv2d()` method, one can do this by
  `keep_img_grad_at_invalid=False`.

* `edge_kernel`: an extra kernel provided for Explict Boundary
  Handling ([paper](https://arxiv.org/abs/1805.03106)), which has the same shape
  as `weight`.
  It is usually `None` for
  forward image filtering, meaning that `weight` is used for both valid (
  interior)
  and invalid (boundary) pixels.

* `optimized_for`: specifies whether the implementation is optimized for
  `'speed'` or `'memory'`. Depending mainly on the image shape, `'speed'` can
  make
  the code run faster by 20~40% at the cost of 50%~100% more GPU memory;
  the memory required by `'memory'` is close to normal padding, so it is
  recommended at a development stage.

**Limitation**: though our method is compatible with stride and dilation, the
current implementation assumes `stride=1` and `dilation=1`. If you need
this extension, please contact us.

### Part II. Convolutional layers

If you are using the following `PyTorch` class in your CNNs,

```python
layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                        stride=1, padding='same', dilation=1, groups=any_groups,
                        bias=any_bias, padding_mode=any_padding_mode) 
```

you can then switch to our method by

```python
from diff_conv2d.layers import DiffConv2dLayer

layer = DiffConv2dLayer(in_channels, out_channels, kernel_size,
                        groups=any_groups, bias=any_bias,
                        keep_img_grad_at_invalid=True, train_edge_kernel=False,
                        optimized_for='memory')
```

The additional arguments are explained below:

* `keep_img_grad_at_invalid`: has the same meaning as above.
* `train_edge_kernel`: specifies whether to activate Explict Boundary
  Handling ([paper](https://arxiv.org/abs/1805.03106)) by training an extra edge
  kernel. If `train_edge_kernel=True`,
  your layer will contain twice the weights as
  a normal `torch.nn.Conv2d` layer has, one for interior and the other for
  boundary pixels.
* `optimized_for`: has the same meaning as above.

**Limitation**: again, the current implementation assumes `stride=1`
and `dilation=1`. If you need
this extension, please contact us.

## Experiments

If you wish to reproduce our experiments, please follow these steps:

1. Move all the files and folders from `experiments/` to the directory
   including `diff_conv2d/` (such as the root of this repo).
2. Install dependencies for experiments:

  ```bash
  pip install -r exp_requirements.txt
  ```

3. Refer to `exp_*_readme.md` for detailed guidance on each experiment, 
including data downloading, training and plotting.

## Acknowledgement

This work is supported by the EPSRC grant, Blueprinting for AI for Science
at Exascale (BASE-II, EP/X019918/1), which is Phase II of the Benchmarking
for AI for Science at Exascale (BASE) grant.

 