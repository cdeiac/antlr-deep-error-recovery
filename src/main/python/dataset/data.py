import json
from types import SimpleNamespace

import torch

from dataset.token import Token


def load_benchmark_data(path_to_train_data, path_to_valid_data):
    with open(path_to_train_data, 'r') as f:
        f.seek(0)
        train = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    with open(path_to_valid_data, 'r') as f:
        f.seek(0)
        test = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    for i in train:
        y_train.append(i.original_data[0])
        X_train.append(i.noisy_data[0])

    for i in test:
        y_valid.append(i.original_data[0])
        X_valid.append(i.noisy_data[0])

    #X_train = X_train[:100]
    #y_train = y_train[:100]
    #X_valid = X_valid[:20]
    #y_valid = y_valid[:20]
    return X_train, y_train, X_valid, y_valid


def build_vocab(train_input, train_target, test_input, test_target):
    corpus = set()
    fill_corpus(corpus, train_input)
    fill_corpus(corpus, train_target)
    fill_corpus(corpus, test_input)
    fill_corpus(corpus, test_target)
    print(corpus)
    token2idx = {'SOS': 0, 'EOS': 1, 'UNK': 2, 'PAD': 3}
    idx2token = {0: 'SOS', 1: 'EOS', 2: 'UNK', 3: 'PAD'}
    for word in corpus:
        if word not in token2idx:
            token2idx[word] = len(token2idx)
            idx2token[len(idx2token)] = word
    print(token2idx)
    print(idx2token)
    print(len(token2idx))
    print(len(idx2token))
    return corpus, token2idx, idx2token


def fill_corpus(vocabulary, sourcefiles):
    for sourcefile in sourcefiles:
        for token in sourcefile.split():
            if token not in vocabulary:
                vocabulary.add(token)


def vectorize(lookup, input):
    #return [[lookup[token] for token in sent.split()] for sent in input]
    result = []
    for sent in input:
        tokenized_sent = []
        for token in sent.split():
            vectorized = Token.UNK
            if token in lookup:
                vectorized = lookup[token]
            tokenized_sent.append(vectorized)
        result.append(tokenized_sent)
    return result



def generate_rolling_windows(x, y, target_dim=0, window_size=128, step_size=64, pad_value=3):
    # select the longer sequence
    pad_len = window_size - (max(x.shape[0], y.shape[0])) % window_size
    # pad sequence to match window size
    x = torch.nn.functional.pad(input=x, pad=(0, pad_len), mode='constant', value=pad_value)
    # generate sliding windows
    return x.unfold(target_dim, window_size, step_size), y.unfold(target_dim, window_size, step_size)


def to_tensor(list_data, device):
    return [torch.tensor(i).long().to(device) for i in list_data]


'''
def window_and_pad_sequences(source, target, window_size, stride):
    # Find the maximum sequence length in the pair
    max_length = max(max(seq.size(0) for seq in source), max(seq.size(0) for seq in target))

    # Initialize empty lists for windowed sequences
    windowed_source = []
    windowed_target = []

    # Create windows with the specified stride
    for i in range(len(source)):
        current_source = source[i]
        current_target = target[i]

        for j in range(0, current_source.size(0) - window_size + 1, stride):
            window = current_source[j:j + window_size]

            # Add start of sequence token for subsequent windows
            if j > 0:
                window = torch.nn.functional.pad(window.unsqueeze(0), (1, 0)).squeeze(0)  # Add one element at the beginning

            windowed_source.append(window)
            windowed_target.append(current_target[j:j + window_size])

    # Pad sequences to be divisible by window_size
    pad_length_source = (window_size - (max_length % window_size)) % window_size
    pad_length_target = (window_size - (max_length % window_size)) % window_size

    padded_source = [torch.nn.functional.pad(seq.unsqueeze(0), (0, pad_length_source)).squeeze(0) for seq in windowed_source]
    padded_target = [torch.nn.functional.pad(seq.unsqueeze(0), (0, pad_length_target)).squeeze(0) for seq in windowed_target]

    return padded_source, padded_target
'''

def prepare_windows(source_list, target_list, window_size, stride, device):
    w_source = []
    w_target = []
    p_target = []
    for i in range(len(source_list)):
        win_s, win_t, pad_t = window_and_pad_sequences(source_list[i], target_list[i], window_size, stride, device)
        w_source.append(win_s)
        w_target.append(win_t)
        p_target.append(pad_t)
    return w_source, w_target, p_target

def window_and_pad_sequences(source, target, window_size, stride, device, padding_token=3):
    source_without_symbols = source[1:]
    target_without_symbols = target[1:]
    #window_size = max(1, window_size-1)
    #stride = max(1, stride-1)
    #window_size = window_size-2
    #if stride != 1:
    #stride = stride-2
    max_length = max(source_without_symbols.size(0), target_without_symbols.size(0))

    pad_length = calculate_padding(max_length, window_size, stride)
    pad_length_x = pad_length + max_length - source_without_symbols.size(0)
    pad_length_y = pad_length + max_length - target_without_symbols.size(0)
    # add padding
    padded_source = torch.nn.functional.pad(source_without_symbols, (0, pad_length_x), mode="constant", value=padding_token)
    padded_target = torch.nn.functional.pad(target_without_symbols, (0, pad_length_y), mode="constant", value=padding_token)
    padded_target_with_eos = torch.nn.functional.pad(target, (0, (pad_length_y - (window_size - stride))), mode="constant", value=padding_token)
    # crate windows
    source_windowed = padded_source.unfold(0, window_size, stride)
    target_windowed = padded_target.unfold(0, window_size, stride)
    #source_windowed = torch.cat((torch.full((source_windowed.size(0), 1), 0, device=device), source_windowed), dim=1)
    target_windowed = torch.cat((torch.full((target_windowed.size(0), 1), 0, device=device), target_windowed), dim=1)
    #source_windowed = torch.cat((source_windowed, torch.full((source_windowed.size(0), 1), 1, device=device)), dim=1)
    #target_windowed = torch.cat((target_windowed, torch.full((target_windowed.size(0), 1), 1, device=device)), dim=1)
    #print(source_windowed)
    return source_windowed, target_windowed, padded_target_with_eos


def calculate_padding(sequence_length, window_size, stride):
    remaining_items = (sequence_length - window_size) % stride
    padding_needed = (stride - remaining_items) % stride
    return padding_needed + (window_size - stride)


def max_pad_tensors(list1, list2, list3, list4):
    # Combine all tensors from the input lists
    all_tensors = list1 + list2 + list3 + list4

    # Find the longest tensor across all lists
    max_length = max(len(tensor) for tensor in all_tensors)
    print(f"max length: {max_length}")

    # Pad all tensors in all lists to the length of the longest tensor
    padded_list1 = [torch.cat((tensor, torch.full((max_length - len(tensor),), 3))) for tensor in list1]
    padded_list2 = [torch.cat((tensor, torch.full((max_length - len(tensor),), 3))) for tensor in list2]
    padded_list3 = [torch.cat((tensor, torch.full((max_length - len(tensor),), 3))) for tensor in list3]
    padded_list4 = [torch.cat((tensor, torch.full((max_length - len(tensor),), 3))) for tensor in list4]

    # Return the padded lists in the same order
    return padded_list1, padded_list2, padded_list3, padded_list4


def pad_tensors_to_max_length(list_of_tensors1, list_of_tensors2, padding_value=3):
    padded_list1 = []
    padded_list2 = []

    for tensor1, tensor2 in zip(list_of_tensors1, list_of_tensors2):
        # Find the maximum length
        max_length = max(len(tensor1), len(tensor2))

        # Pad the tensors to the maximum length
        tensor1_padded = torch.nn.functional.pad(tensor1, pad=(0, max_length - len(tensor1)), mode='constant', value=padding_value)
        tensor2_padded = torch.nn.functional.pad(tensor2, pad=(0, max_length - len(tensor2)), mode='constant', value=padding_value)

        padded_list1.append(tensor1_padded)
        padded_list2.append(tensor2_padded)

    return padded_list1, padded_list2
