import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='PyTorch MixMatch Example')
parser.add_argument('--data', default='0', type=str, choices=['TIN', 'LSUN', 'Gau', 'Uni', 'Clean'],
                    help='type of OOD data')
parser.add_argument('--n-labeled', default=250, type=int, choices=[250, 1000, 4000],
                    help='number of labeled data')
parser.add_argument('--method', default='proposed', type=str, choices=['baseline', 'proposed'],
                    help='choice of method')
parser.add_argument('--gpu', default='0', type=str,
                    help='id for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

dataset = {'TIN': 'Imagenet_resize', 'LSUN': 'LSUN_resize', 'Gau':'Gaussian', 'Uni':'Uniform', 'Clean': 'none'}

if args.method == 'proposed':
    subprocess.run(f"python train_multi.py --gpu {args.gpu} --n-labeled {args.n_labeled} --out runs_proposed/cifar10_{args.data}@{args.n_labeled} --outdata {dataset[args.data]}", shell=True)
elif args.method == 'baseline':
    subprocess.run(f"python train.py --gpu {args.gpu} --n-labeled {args.n_labeled} --out runs_baseline/cifar10_{args.data}@{args.n_labeled} --outdata {dataset[args.data]}", shell=True)
else:
    raise ValueError("Wrong method!")