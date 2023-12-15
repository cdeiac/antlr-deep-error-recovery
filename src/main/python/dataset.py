import json
from types import SimpleNamespace
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


SOS_TOKEN: str = 'SOS'
EOS_TOKEN: str = 'EOS'
UNK_TOKEN: str = 'UNK'
SOS_IDX: int = 0
EOS_IDX: int = 1
UNK_IDX: int = 2


class Dataset(Dataset):
    def __init__(self, filename):
        self.original_data = []
        self.noisy_data = []
        self.token2index = {SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}
        self.index2token = {SOS_IDX: SOS_TOKEN, EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}
        self.n_token = 3
        self.load_json_data(filename)
        #self.encode_data()
        #self.generate_splits()
        #self.to_tensor()

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        noisy = self.noisy_data[idx]
        original = self.original_data[idx]
        return {'noisy': noisy, 'original': original}

    def load_json_data(self, filename):
        with open(filename, 'r', encoding="utf8") as f:
            f.seek(0)
            json_data = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        # TODO: Do this part after split and only add train & validation data to vocabulary!
        for data in json_data:
            len_orig = len(data.source.source)
            len_nois = len(data.source.sourceWithNoise)
            self.original_data.append(self.enrich_with_control_tokens(data.source.source))#, (len_nois - len_orig)))
            self.noisy_data.append(self.enrich_with_control_tokens(data.source.sourceWithNoise))#, 0))

    def enrich_with_control_tokens(self, sourcefile):#, padding_tokens):
        #self.add_sourcefile(sourcefile)
        file = f'{SOS_TOKEN} {sourcefile} {EOS_TOKEN}'#self.add_padding(sourcefile, padding_tokens) + " EOS"
        self.add_sourcefile(file)
        return file

    def add_padding(self, sourcefile, amount):
        if amount > 0:
            for i in range(amount):
                sourcefile += " PAD"
        return sourcefile

    def add_sourcefile(self, sourcefile):
        for token in sourcefile.split():
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_token
            self.index2token[self.n_token] = token
            self.n_token += 1

    def encode_data(self):
        self.original_data = [self.encode(sourcefile) for sourcefile in self.original_data]
        self.noisy_data = [self.encode(sourcefile) for sourcefile in self.noisy_data]

    def generate_splits(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.noisy_data,
                                                                                self.original_data,
                                                                                test_size=0.2,
                                                                                random_state=1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.25,
                                                                              random_state=1)

    def encode(self, sourcefile):
        return [self.get_by_token(token) for token in sourcefile.split()]

    def decode(self, sourcefile):
        return [self.get_by_index(idx) for idx in sourcefile.split()]

    def get_by_token(self, token):
        try:
            return self.token2index[token]
        except KeyError:
            return UNK_TOKEN

    def get_by_index(self, idx):
        try:
            return self.index2token[idx]
        except KeyError:
            return UNK_IDX

    def source_file_to_ids(self, source_file, vocab):
        return [vocab[token] for token in source_file]

    def ids_to_source_file(self, id_list, vocab):
        return [vocab[id] for id in id_list]

    def to_tensor(self):
        for i, data in enumerate(self.original_data):
            self.original_data[i] = torch.tensor(data)
        for i, data in enumerate(self.noisy_data):
            self.noisy_data[i] = torch.tensor(data)
