import logging
import os
import random

import torch
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from config import parse_config
from dataset import Dataset
from utils.data import dump_data_and_cache


class Runner:
    def __init__(self):
        # setup
        self.config = self.__init_config()
        self.__init_cache()
        self.__init_logger()
        self.__set_seed()

    def load_dataset(self):
        logging.info('Loading JSON data. . .')
        java_dataset = Dataset(self.config.data_path)
        return java_dataset

    def split_dataset(self, dataset: Dataset):
        logging.info('Splitting train and test data. . .')
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        logging.info(f'Dataset size: {len(dataset)}, split sizes: train={train_size}, test={test_size}')
        train_data, test_data = random_split(dataset,
                                             [train_size, test_size],
                                             generator=Generator().manual_seed(self.config.seed))
        #dump_json(self.get_filepath_for_test(), format_data_for_dump(test_data))
        dump_data_and_cache(self.config, 'test', test_data)
        #return train_data, test_data

    def generate_cv_folds(self, train_dataset: Dataset):
        logging.info('Generating folds. . .')
        # create fold
        kfold = KFold(n_splits=3, shuffle=True, random_state=self.config.seed)
        # split train and test data
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):
            # Sample at random from a given list of ids, no replacement.
            train_subsampler = SubsetRandomSampler(train_ids)
            valid_subsampler = SubsetRandomSampler(valid_ids)
            # Define data loaders for training and validation data in this fold
            train_loader = DataLoader(train_dataset, sampler=train_subsampler)    # batch_size=1
            valid_loader = DataLoader(train_dataset, sampler=valid_subsampler)      # batch_size=1
            dump_data_and_cache(self.config, 'train', train_loader, fold)
            dump_data_and_cache(self.config, 'validation', valid_loader, fold)
            logging.info(f'Fold {fold} completed: train size={len(train_loader)}, validation size={len(valid_loader)}')

    def compile_model(self, vocab_size: int):
        logging.info('Compiling model. . .')
        model = Seq2Seq(vocab_size,
                        self.config.embed_dim,
                        self.config.hidden_dim,
                        self.config.num_of_layers,
                        vocab_size,
                        self.config.bidirectional)
        return model #torch.compile(model) TODO: Uncomment!

    def __set_seed(self):
        logging.info(f'Setting seed to {self.config.seed}')
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        # for deterministic behavior or RNNs
        #torch.use_deterministic_algorithms(True)

    @staticmethod
    def __init_config():
        logging.info('Parsing config. . .')
        config = parse_config()
        logging.info(f'Parsed config: {config.__dict__}')
        return config

    def __init_logger(self):
        logging.info(f'Initializing logging directory: {self.config.log_dir}')
        os.mkdir(self.config.log_dir)
        os.mkdir(self.config.model_dir)
        logging.info('Init logger. . .')
        logging.basicConfig(filename=f'{self.config.log_dir}{self.config.model_name}.log',
                            filemode='a',
                            format='%(asctime)s |   %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)

    def __init_cache(self):
        logging.info(f'Initializing cache directory: {self.config.cache_dir}')
        os.mkdir(self.config.cache_dir)

