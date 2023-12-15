import glob
from pathlib import Path
import json
from typing import Any

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class Dataloader(Dataset):

    def __init__(self, path: str):
        self.vocabulary = None
        self.label_encoder = LabelEncoder()
        self.x_data = []
        self.y_data = []
        self.x_enc = []
        self.y_enc = []
        self.data = self.load_data(path)
        [self.x_data.append(d['source']['sourceWithNoise']) for d in self.data]
        [self.y_data.append(d['source']['source']) for d in self.data]
        # max length
        self.max_length = 0
        for sequence in self.data:
            seq_len = len(sequence)
            if seq_len > self.max_length:
                self.max_length = seq_len
        self.build_vocabulary()

    def load_data(self, path: str):
        with open(path, 'r') as file:
            return json.load(file)

    def build_vocabulary(self):
        for x in self.x_data:
            self.x_enc.append(self.label_encoder.fit_transform(x.split(' ')))
        for y in self.y_data:
            self.y_enc.append(self.label_encoder.fit_transform(y.split(' ')))
        self.vocabulary = self.label_encoder.classes_

    def __len__(self) -> int:
        return len(self.data)

    def __seq_len__(self) -> int:
        return self.max_length

    def __getitem__(self, index: int) -> Any:
        data = self.data[index]
        return data