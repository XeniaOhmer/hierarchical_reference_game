import torch
from dataset import AnyItemSet
import argparse
import os

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default=None)
parser.add_argument('--dimensions', nargs="*", type=int, default=[5, 5, 5],
                    help='Number of features for every perceptual dimension')
parser.add_argument('--intentional_distractors', type=int, default=10,
                    help='Number of intentional distractor objects for the receiver')
parser.add_argument('--sample_scaling', type=int, default=10)
parser.add_argument('--upsample', type=bool, default=True,
                    help='If upsampling is used, sampling of intention vectors is corrected such each level of the'
                         'conceptual hierarchy is equally likely. Otherwise intentions are sampled (N over k), where'
                         'k is the number of same/any and N the number of dimensions')
parser.add_argument('--balanced_distractors', type=bool, default=False,
                    help='corrects for equal probability of all levels of abstraction in distractors')

args = parser.parse_args()


item_set = AnyItemSet(args.dimensions,
                      distractors=args.intentional_distractors,
                      sample_scaling=args.sample_scaling,
                      upsample=args.upsample,
                      balanced_distractors=args.balanced_distractors,
                      random_distractors=0)

if not os.path.exists('data/'):
    os.makedirs('data/')

if args.path is None:
    path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ').ds')
else:
    path = args.path

with open(path, "wb") as f:
    torch.save(item_set, f)

print("Data set is saved as: " + path)
