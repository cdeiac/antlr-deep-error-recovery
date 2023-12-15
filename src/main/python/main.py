import random

import torch.optim as optim

from data.data import create_complex_data, build_vocab, vectorize, to_tensor
from models.base import DenoisingAutoEncoder
from training import train_one_epoch, evaluate_one_epoch
import numpy as np
import torch
import torch.nn as nn

#if __name__ == '__main__':
    #json_data = json.load(open(os.path.dirname(os.path.realpath(__file__)).replace('python', 'resources/generated/noisy_jhetas_clean_10.json')))
    #json_arr = []
    #for i, item in enumerate(json_data):
    #    if i < 5000:
    #        json_arr.append(item)

    #print(len(json_arr))
    #with open("../resources/generated/data.json", "w+") as f:
    #    json.dump(json_arr, f)



if __name__ == '__main__':
    # config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load data
    train_input, train_target, test_input, test_target = create_complex_data()
    # create vocabulary
    corpus, token2index, index2token = build_vocab(train_input, train_target, test_input, test_target)
    # vectorize
    train_batch = vectorize(token2index, train_input)
    train_target_batch = vectorize(token2index, train_target)
    test_batch = vectorize(token2index, test_input)
    test_target_batch = vectorize(token2index, test_target)
    # to numpy array
    train_batch = np.array(train_batch)
    train_target_batch = np.array(train_target_batch)
    test_batch = np.array(test_batch)
    test_target_batch = np.array(test_target_batch)
    # to tensor
    train_batch = to_tensor(train_batch, device)
    train_target_batch = to_tensor(train_target_batch, device)
    test_batch = to_tensor(test_batch, device)
    test_target_batch = to_tensor(test_target_batch, device)

    # training config
    SEED = 11
    EPOCHS = 10
    VOCAB_SIZE = len(token2index)
    EMBED_DIM = 128
    LIN1_SIZE = 32
    HID_DIM = 64
    N_LAYERS = 2
    BIDIRECTIONAL = False
    loss_function = nn.CrossEntropyLoss(ignore_index=3)
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = DenoisingAutoEncoder(VOCAB_SIZE, EMBED_DIM, HID_DIM, LIN1_SIZE, N_LAYERS, VOCAB_SIZE, device, BIDIRECTIONAL)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=3,
                                                           threshold=0.01,
                                                           threshold_mode='abs',
                                                           verbose=True)
    print(model)

    # training
    epoch_train_loss = []
    epoch_valid_loss = []
    for epoch in range(EPOCHS):
        epoch_train_loss.append(train_one_epoch(train_batch,
                                                train_target_batch,
                                                model,
                                                optimizer,
                                                loss_function))
        epoch_valid_loss.append(evaluate_one_epoch(test_batch,
                                                   test_target_batch,
                                                   model,
                                                   loss_function,
                                                   scheduler,
                                                   index2token))
        print('---------------------')
    print(f'Train Loss: {epoch_train_loss}\nValid Loss: {epoch_valid_loss}')
