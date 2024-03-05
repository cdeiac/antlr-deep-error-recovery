import torch
from torch import optim, nn

from config import Config
from dataset.token import Token
from models.lstm import LSTMDenoiser


class TrainingSession:

    def __init__(self, config: Config, vocab_size):
        self.max_epochs = 10
        self.vocab_size = vocab_size
        self.embedding_dim = 128
        self.hidden_size = 128
        self.num_layers = 1
        self.bidirectional = True
        self.config = config
        self.start_fold,  self.loss_function, self.model, self.optimiser, self.scheduler = self.init_session()

    def save_checkpoint(self, current_fold: int):
        torch.save(self.model.state_dict(), f"{self.config.checkpoint_dir}/checkpoint{current_fold}.pt")

    def init_session(self):
        model = self.__init_model()
        start_fold = 0
        if self.config.load_checkpoint:
            model.load_state_dict(torch.load(self.config.checkpoint_path, map_location=self.config.device))
            start_fold = int(self.config.checkpoint_path[-4])
        optimiser = self.__init_optimiser(model)
        scheduler = self.__init_scheduler(optimiser)
        loss_function = nn.CrossEntropyLoss(ignore_index=Token.PAD)
        return start_fold, loss_function, model, optimiser, scheduler

    def __init_model(self):
        model = LSTMDenoiser(self.vocab_size, self.embedding_dim, self.hidden_size, self.num_layers, self.bidirectional)
        model.to(self.config.device)
        return model

    def __init_optimiser(self, model):
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    def __init_scheduler(self, optimiser):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode='max',
            factor=0.1,
            patience=3,
            threshold=0.001,
            threshold_mode='abs'
        )


