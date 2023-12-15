import json
import pickle
from typing import Optional

from torch.utils.data import DataLoader

from config import Config


def load_json(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(filename: str, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_cache(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def dump_cache(filename: str, dataloader: DataLoader):
    with open(filename, 'wb') as f:
        pickle.dump(dataloader, f)


def load_from_cache(self, datatype: str, fold_id: Optional[int] = None) -> DataLoader:
    # TODO: Load to device?
    if datatype != 'test':
        cache = load_cache(self.get_filepath_for_fold_cache(fold_id, datatype))
        return cache
    else:
        return load_cache(self.get_filepath_for_test_cache())


def dump_data_and_cache(config: Config, datatype: str, dataloader: DataLoader, fold_id: Optional[int] = None):
    if datatype != 'test':
        # train, validation dataset
        dump_json(get_filepath_for_fold(config, fold_id, datatype), format_data_for_dump(dataloader))
        dump_cache(get_filepath_for_fold_cache(config, fold_id, datatype), dataloader)
    else:
        # test dataset
        dump_json(get_filepath_for_test(config), format_data_for_dump(dataloader))
        dump_cache(get_filepath_for_test_cache(config), dataloader)


def format_data_for_dump(dataloader: DataLoader):
    dump = []
    for data in dataloader:
        dump.append({
            "original_data": data['original'],
            "noisy_data": data['noisy']
            #"cv_indices": list(dataloader.sampler.indices)
        })
    return dump


def get_filepath_for_fold(config: Config, fold_id: int, datatype: str):
    return f'{config.log_dir}fold_{fold_id}_{datatype}.json'


def get_filepath_for_fold_cache(config: Config, fold_id: int, type: str):
    return f'{config.cache_dir}fold_{fold_id}_{type}.pickle'


def get_filepath_for_test(config: Config):
    return f'{config.log_dir}test.json'


def get_filepath_for_test_cache(config: Config):
    return f'{config.cache_dir}test.pickle'

