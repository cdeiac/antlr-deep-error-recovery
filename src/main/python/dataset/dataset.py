import json
from types import SimpleNamespace

import numpy as np
from torch.utils.data import Dataset

from config import Config
from dataset.token import Token

SOS_TOKEN: str = 'SOS'
EOS_TOKEN: str = 'EOS'
PAD_TOKEN: str = 'PAD'
index2token_path = 'src/main/python/persistent/index2token.json'
token2index_path = 'src/main/python/persistent/token2index.json'


class DatasetLoader(Dataset):
    def __init__(self, config: Config):
        self.original_data = []
        self.noisy_data = []
        self.noise_operations = []
        self.token2index = {SOS_TOKEN: Token.SOS, EOS_TOKEN: Token.EOS, PAD_TOKEN: Token.PAD}
        self.index2token = {Token.SOS: SOS_TOKEN, Token.EOS: EOS_TOKEN, Token.PAD: PAD_TOKEN}
        self.n_token = 3
        self.load_json_data(config)
        # load static vocabularies to be consistent across all trials
        self.load_vocabs()

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        return {
            'original': self.original_data[idx],
            'noisy': self.noisy_data[idx],
            'noise_operations': self.noise_operations[idx]
        }

    def sub_sample_original(self, idx):
        for i in idx:
            yield self.original_data[i]

    def sub_sample_noisy(self, idx):
        for i in idx:
            yield self.noisy_data[i]

    def sub_sample_nops(self, idx):
        for i in idx:
            yield self.noise_operations[i]

    def load_json_data(self, config: Config):
        if config.load_cv is False:
            with open(config.data_path, 'r', encoding="utf8") as f:
                json_data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
            for data in json_data:
                self.original_data.append(self.enrich_with_control_tokens(data.source.source))
                self.noisy_data.append(self.enrich_with_control_tokens(data.source.sourceWithNoise))
                self.noise_operations.append(data.source.noiseOperations)

    def enrich_with_control_tokens(self, sourcefile):
        return f'{SOS_TOKEN} {sourcefile} {EOS_TOKEN}'

    def load_vocabs(self):
        with open(index2token_path, 'r') as json_file:
            self.index2token = json.load(json_file)

        with open(token2index_path, 'r') as json_file:
            self.token2index = json.load(json_file)
