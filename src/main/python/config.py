import argparse
import time
from argparse import Namespace

import torch
from torch import device

CACHE_BASE_DIR: str = './cache/'
LOG_BASE_DIR: str = './logs/'
MODEL_DIR: str = './saved_models/'
SEED: int = 11
FOLDS: int = 3
DEVICE: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_config():
    parser = argparse.ArgumentParser(description='Required command line arguments')
    # data
    parser.add_argument('--data_path', type=str, help='The path to the data file')
    # model
    parser.add_argument('--model_name', type=str, help='The name of the model')
    parser.add_argument('--embed_dim', type=int, help='The embedding dimension of the model')
    parser.add_argument('--hidden_dim', type=int, help='The hidden dimension of the model')
    parser.add_argument('--num_of_layers', type=int, help='The number of LSTM layers of the model')
    parser.add_argument('--bidirectional', type=bool, help='Whether or not the model is bidirectional')
    # training
    parser.add_argument('--device', type=str, help='The computation device (CPU or CUDA)')
    parser.add_argument('--log_dir', type=str, help='The logging directory')
    parser.add_argument('--model_dir', type=str, help='The directory for the saved models')
    parser.add_argument('--epochs',
                        nargs='?',
                        const=100,
                        type=int,
                        help='The number of epochs to train the model for')
    parser.add_argument('--batch_size',
                        nargs='?',
                        const=1,
                        type=int,
                        help='The batch size for the dataloader')
    parser.add_argument('--lr_start_value',
                        nargs='?',
                        const=0.01,
                        type=float,
                        help='The starting value of the learning rate')
    parser.add_argument('--lr_factor',
                        type=float,
                        help='The factor by which the learning rate is adjusted')
    # parse arguments
    args = parser.parse_args()
    return Config(args)


class Config:
    def __init__(self, args: Namespace):
        timestamp = int(time.time())
        self.data_path = args.data_path
        self.model_name = args.model_name
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.num_of_layers = args.num_of_layers
        self.bidirectional = args.bidirectional
        self.device = DEVICE
        self.lr_start_value = args.lr_start_value
        self.lr_factor = args.lr_factor
        self.log_dir = f'{LOG_BASE_DIR}{timestamp}/'
        self.model_dir = f'{MODEL_DIR}{timestamp}/'
        self.cache_dir = f'{CACHE_BASE_DIR}{timestamp}/'
        self.seed = SEED
        self.epochs = args.epochs
        self.folds = FOLDS
