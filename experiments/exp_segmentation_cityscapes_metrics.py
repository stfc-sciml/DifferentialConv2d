import argparse

import matplotlib.pyplot as plt
import torch

from exp_utils import conv2d_methods

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='random seed')
    args = parser.parse_args()
    metrics = ['iou_score', 'f1_score', 'accuracy']
    print(' '.join(['method'] + metrics))

    plt.figure()
    for method in conv2d_methods.keys():
        try:
            hist = torch.load(
                f'./exp_results/cityscapes/{method}_seed{args.seed}.hist.pt')
            values = [f'{hist[m] * 100:.1f}%' for m in metrics]
            print(' '.join([method] + values))
            plt.plot(hist['epoch_losses'], label=method)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
        except:
            pass
    plt.legend()
    plt.show()
