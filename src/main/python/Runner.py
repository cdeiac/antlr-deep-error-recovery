import gc
import logging
import os
import random

import torch
from sklearn.model_selection import KFold, train_test_split

from config import parse_config
from utils.training import predict, train_one_epoch, evaluate_one_epoch
from utils.data import dump_data_and_cache, load_cache, format_scores, save_training_logs
from dataset.dataset import DatasetLoader
from utils.training_session import TrainingSession


class Runner:
    def __init__(self):
        # setup
        self.vocab_size = 0
        self.config = self.__init_config()
        self.__init_cache()
        self.logger = self.__init_logger()
        self.__set_seed()
        self.__load_dataset()
        self.training_session = TrainingSession(self.config, self.vocab_size)

    def train_model(self):
        self.logger.info("Starting training. . .")
        logs = {}
        for fold_id in range(self.training_session.start_fold, 3):
            self.logger.info(f"Training on fold: {fold_id}")
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            test_accs = []
            # load all data from cache
            train_target, train_source, val_target, val_source, test_target, test_source = \
                load_cache(self.config, fold_id)
            # run max_epochs per fold
            for epoch in range(1, self.training_session.max_epochs+1):
                train_loss, train_acc = train_one_epoch(train_source, train_target, self.training_session)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_loss, val_acc = evaluate_one_epoch(val_source, val_target, self.training_session)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                self.logger.info(f"Epoch: {epoch}/{self.training_session.max_epochs} | "
                                 f"train_loss: {train_loss} train_acc: {train_acc}  | "
                                 f"val_loss: {val_loss} val_acc: {val_acc}")
            # evaluate on test set after max epochs
            test_acc = predict(test_source, test_target, self.training_session)
            test_accs.append(test_acc)
            self.logger.info(f"Training fold completed, test_acc: {test_acc}")
            self.logger.info("Save checkpoints and logs. . .")
            self.training_session.save_checkpoint(fold_id)
            logs[fold_id] = format_scores(train_losses, train_accs, val_losses, val_accs, test_accs)
            save_training_logs(self.config, logs)
        self.logger.info("Training fully completed!")

    def load_cv_fold(self, fold_id: int):
        train_original, train_noisy, validation_original, validation_noisy, test_original, test_noisy = \
            load_cache(self.config, fold_id)
        return train_original, train_noisy, validation_original, validation_noisy, test_original, test_noisy

    def generate_cv_folds(self, dataset: DatasetLoader):
        self.logger.info('Generating folds. . .')
        # create fold
        kfold = KFold(n_splits=3, shuffle=True, random_state=self.config.seed)
        # split train and test data
        for fold, (train_idx, test_ids) in enumerate(kfold.split(dataset)):
            test_original, test_noisy, test_nops = dataset.sub_sample(test_ids)
            train_original, train_noisy, train_nops = dataset.sub_sample(train_idx)
            train_original, val_original = train_test_split(train_original)
            train_noisy, val_noisy = train_test_split(train_noisy)
            train_nops, val_nops = train_test_split(train_nops)
            self.logger.info(
                f'Fold {fold} completed: train size={len(train_noisy)}, '
                f'validation size={len(val_noisy)}, '
                f'test size={len(test_noisy)}'
            )
            dump_data_and_cache(self.config, dataset, 'train', train_original, train_noisy, train_nops, fold)
            self.logger.info(f'Fold {fold} train cache initialized')
            dump_data_and_cache(self.config, dataset, 'validation', val_original, val_noisy, val_nops, fold)
            self.logger.info(f'Fold {fold} validation cache initialized')
            dump_data_and_cache(self.config, dataset, 'test', test_original, test_noisy, test_nops, fold)
            self.logger.info(f'Fold {fold} test cache initialized')
            # cleanup references
            gc.collect()
            del test_original, test_noisy, test_nops, \
                train_original, train_noisy, train_nops, \
                val_original, val_noisy, val_nops

    def split_train_validation_data(self, data: [], ):
        return train_test_split(data, 0.1, shuffle=False)

    def __set_seed(self):
        self.logger.info(f'Setting seed to {self.config.seed}')
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        # for deterministic behavior or RNNs
        #torch.use_deterministic_algorithms(True)

    def __init_config(self):
        return parse_config()

    def __init_logger(self):
        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        logger = logging.getLogger(__name__)
        logfile_name = f'{self.config.log_dir}/{self.config.model_name}.log'
        with open(logfile_name, 'a'): pass
        file_handler = logging.FileHandler(logfile_name)
        formatter = logging.Formatter(fmt='%(asctime)s |   %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(f'Initializing logger in directory: {self.config.log_dir}')
        return logger

    def __init_cache(self):
        logging.info(f'Initializing cache directory: {self.config.cache_dir}')
        if not os.path.exists(self.config.cache_dir):
            os.mkdir(self.config.cache_dir)

        logging.info(f'Initializing cv directory: {self.config.cv_dir}')
        if not os.path.exists(self.config.cv_dir):
            os.mkdir(self.config.cv_dir)

    def __load_dataset(self):
        self.logger.info('Loading JSON data. . .')
        dataset = DatasetLoader(self.config)
        self.vocab_size = len(dataset.token2index)
        if not self.config.load_cv:
            self.generate_cv_folds(dataset)
