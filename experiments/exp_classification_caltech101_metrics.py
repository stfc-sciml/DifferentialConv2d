import argparse

import matplotlib.pyplot as plt
import torch

from exp_utils import conv2d_methods

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='random seed')
    args = parser.parse_args()

    print('method accuracy')
    plt.figure()
    for method in conv2d_methods.keys():
        try:
            hist = torch.load(
                f'./exp_results/caltech101/{method}_seed{args.seed}.hist.pt')
            print(method, f'{hist["accuracy"] * 100:.1f}%')
            plt.plot(hist['epoch_losses'], label=method)
        except:
            pass
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
