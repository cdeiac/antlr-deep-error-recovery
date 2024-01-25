import json
import pickle
from typing import Optional

import torch

from config import Config
from dataset.data import pad_tensors_to_max_length
from dataset.dataset import DatasetLoader


def load_json(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(filename: str, original, noisy, nops):
    with open(filename, 'w') as f:
        for orig, nois, nop in zip(original, noisy, nops):
            json.dump({
                "original_data": orig,
                "noisy_data": nois,
                "noise_operations": nop
            }, f)


def load_cache(config: Config, fold_id: int):
    # training
    with open(get_filepath_for_fold_cache(config, fold_id, 'train'), 'rb') as f:
        data = pickle.load(f)
    train_original = data['original']
    train_noisy = data['noisy']
    # validation
    with open(get_filepath_for_fold_cache(config, fold_id, 'validation'), 'rb') as f:
        data = pickle.load(f)
    val_original = data['original']
    val_noisy = data['noisy']
    # test
    with open(get_filepath_for_fold_cache(config, fold_id, 'test'), 'rb') as f:
        data = pickle.load(f)
    test_original = data['original']
    test_noisy = data['noisy']
    return train_original, train_noisy, val_original, val_noisy, test_original, test_noisy


def dump_cache(filename: str, original: [], noisy: []):
    data = {'original': original, 'noisy': noisy}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def dump_data_and_cache(
        config: Config,
        dataset: DatasetLoader,
        datatype: str,
        original: [],
        noisy: [],
        nops: [],
        fold_id: Optional[int] = None
):
    with open(get_filepath_for_fold(config, fold_id, datatype), 'w') as json_file, \
            open(get_filepath_for_fold_cache(config, fold_id, datatype), 'wb') as pickle_file:
        for orig, nois, nop in zip(original, noisy, nops):
            json.dump({
                "original_data": orig,
                "noisy_data": nois,
                "noise_operations": nop
            }, json_file)
            orig = encode_and_to_tensor(config, orig, dataset.token2index)
            nois = encode_and_to_tensor(config, nois, dataset.token2index)
            # pad both sequences to be of the length
            orig, nois = pad_tensors_to_max_length(nois, orig)
            # dump cache
            pickle.dump({'original': orig, 'noisy': nois}, pickle_file)


def format_data_for_dump(original: [], noisy: [], nops: []):
    dump = []
    for orig, nois, nop in zip(original, noisy, nops):
        dump.append({
            "original_data": orig,
            "noisy_data": nois,
            "noise_operations": nop
        })
    return dump


def save_training_logs(config: Config, logs: {}):
    dump_json(f'{config.log_dir}/scores.json', logs)


def format_scores(
        train_losses: [],
        train_accs: [],
        val_losses: [],
        val_accs: [],
        test_accs: []
):
    return {
        'train_loss': train_losses,
        'train_accs': train_accs,
        'val_loss': val_losses,
        'val_accs': val_accs,
        'test_accs': test_accs
    }


def get_filepath_for_fold(config: Config, fold_id: int, datatype: str):
    return f'{config.cv_dir}/fold_{fold_id}_{datatype}.json'


def get_filepath_for_fold_cache(config: Config, fold_id: int, datatype: str):
    return f'{config.cache_dir}/fold_{fold_id}_{datatype}.pickle'


def encode_and_to_tensor(config: Config, sourcefile: [], token2index: dict):
    encoded_sourcefile = []
    for token in sourcefile.split():
        encoded_sourcefile.append(get_by_token(token, token2index))
    return torch.tensor(encoded_sourcefile, device=config.device).long()


def to_tensor(list_data, device):
    return [torch.tensor(i).long().to(device) for i in list_data]


def get_by_token(token, token2index: dict):
    try:
        return token2index[token]
    except KeyError:
        print(f"Found OOV Word: {token}")
