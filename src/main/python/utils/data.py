import json
import pickle
from typing import Optional

import torch


from config import Config
from dataset.data import pad_tensors_to_max_length
from dataset.dataset import DatasetLoader, UNK_TOKEN
from dataset.token import Token


def load_json(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(filename: str, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f)


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
    dump_json(get_filepath_for_fold(config, fold_id, datatype), format_data_for_dump(original, noisy, nops))
    # encode and create tensors before dumping cache
    original = encode_and_to_tensor(config, original, dataset.token2index)
    noisy = encode_and_to_tensor(config, noisy, dataset.token2index)
    # pad both sequences to be of the length
    original, noisy = pad_tensors_to_max_length(noisy, original)
    dump_cache(get_filepath_for_fold_cache(config, fold_id, datatype), original, noisy)


def format_data_for_dump(original: [], noisy: [], nops: []):
    dump = []
    for i in range(len(original)):
        dump.append({
            "original_data": original[i],
            "noisy_data": noisy[i],
            "noise_operations": nops[i]
        })
    return dump


def save_training_logs(config: Config, logs: {}):
    dump_json(f'{config.log_dir}/scores.json',logs)


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


def encode_and_to_tensor(config: Config, data: [], token2index: dict):
    encoded_data = []
    tensor_data = []
    for i, d in enumerate(data):
        encoded_sourcefile = []
        sourcefile = data[i].split()
        for token in sourcefile:
            encoded_sourcefile.append(get_by_token(token, token2index))
        encoded_data.append(encoded_sourcefile)
        tensor_data.append(torch.tensor(encoded_data[i], device=config.device).long())
    return tensor_data


def to_tensor(list_data, device):
    return [torch.tensor(i).long().to(device) for i in list_data]


def get_by_token(token, token2index: dict):
    try:
        return token2index[token]
    except KeyError:
        print(f"Found OOV Word: {token}")
        return Token.UNK
