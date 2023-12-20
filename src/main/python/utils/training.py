import logging

import torch
from torch import optim, nn

from config import Config
from dataset import Dataset


class TrainingSession:

    def __init__(self, config: Config, model: nn.Module, checkpoint=None):
        self.config = config
        self.model = self.__init_model(model, checkpoint)
        self.optimizer = self.__init_optimizer(config, checkpoint)
        self.scheduler = self.__init_scheduler(config, checkpoint)

    def __init_model(self, model: nn.Module, checkpoint=None):
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
        return model

    def __init_optimizer(self, config: Config, checkpoint=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=config.lr_start_value)
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer

    def __init_scheduler(self, config: Config, checkpoint=None):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               mode='min',
                                                               factor=config.lr_factor,
                                                               patience=2,
                                                               threshold=0.0001,
                                                               threshold_mode='abs')
        if checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        return scheduler
