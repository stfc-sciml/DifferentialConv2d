import argparse

import torch

from exp_utils import conv2d_methods

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='random seed')
    args = parser.parse_args()

    for method in conv2d_methods.keys():
        try:
            hist = torch.load(
                f'./exp_results/etopo/{method}_seed{args.seed}.hist.pt')
            print(method)
            for L in [64, 128, 192, 256]:
                ratio = hist['frame_loss'][L] / hist['inter_loss'][L]
                print(f"{L} {hist['inter_loss'][L] * 10000:.1f} "
                      f"{hist['frame_loss'][L] * 10000:.1f} "
                      f"{ratio * 100:.1f}%")
        except:
            pass
