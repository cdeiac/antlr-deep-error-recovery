import glob
import argparse
import os.path
from argparse import Namespace

import torch


def parse_config():
    parser = argparse.ArgumentParser(description='Required command line arguments')
    # data
    parser.add_argument('--data_dir', type=str, help='The directory to the data file')
    parser.add_argument('--load_cv',
                        nargs='?',
                        const=False,
                        type=bool,
                        help='Whether or not to load existing CV splits to resume training')
    # model
    parser.add_argument('--load_checkpoint',
                        nargs='?',
                        const=False,
                        type=bool,
                        help='Whether to load an existing checkpoint to resume training')
    # parse arguments
    args = parser.parse_args()
    return Config(args)


class Config:
    def __init__(self, args: Namespace):
        self.data_dir = args.data_dir
        self.data_path = f'src/main/resources/generated/{self.data_dir}/noisy_jhetas_clean.json'
        self.model_name = 'lstm_model'
        self.load_checkpoint = args.load_checkpoint
        self.checkpoint_dir = f'src/main/python/data/generated/checkpoints/{self.data_dir}'
        self.checkpoint_path = max(glob.glob(self.checkpoint_dir + "/*"),
                                   key=os.path.getmtime) if self.load_checkpoint else None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = f'src/main/python/logs/{self.data_dir}'
        self.cache_dir = f'src/main/python/data/generated/cache/{self.data_dir}'
        self.cv_dir = f'src/main/python/data/generated/cv/{self.data_dir}'
        self.load_cv = False if args.load_cv is None else True
        self.seed = 11
