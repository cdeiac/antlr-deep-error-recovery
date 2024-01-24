import numpy as np
from matplotlib import pyplot as plt

from Runner import Runner


def init_countmap_of_tensors(tensor_list):
    # Flatten the list of arrays and concatenate them into a single NumPy array
    flattened = [tensor.flatten().numpy() for tensor in tensor_list]
    all_values = np.concatenate(flattened)
    # Use numpy's unique function to get unique values and their counts
    unique_values, counts = np.unique(all_values, return_counts=True)
    return {unique_value: 0 for unique_value in unique_values}
    # Create a dictionary with unique values as keys and their counts as values
    #return dict(zip(unique_values, counts))


def add_counts_exceeds_max_count(tensor, count_map, max_count):
    temp_map = count_map.copy()
    for tensor_value in tensor:
        val = tensor_value.item()
        temp_map[val] = temp_map[val] + 1
    return max(temp_map.values()) > max_count


def add_counts_to_map(tensor, count_map):
    for tensor_value in tensor:
        val = tensor_value.item()
        count_map[val] = count_map[val] + 1
    return count_map


def create_balanced_dataset(source_list, target_list, max_elements, max_count):
    count_map = init_countmap_of_tensors(target_list)
    selected_source_samples = []
    selected_target_samples = []
    for i in range(len(target_list)):
        if len(selected_target_samples) == max_elements:
            break
        if not add_counts_exceeds_max_count(target_list[i], count_map, max_count):
            selected_target_samples.append(target_list[i])
            count_map = add_counts_to_map(target_list[i], count_map)
            selected_source_samples.append(source_list[i])
            if max(count_map.values()) >= max_count:
                break
    return selected_source_samples, selected_target_samples


def resample(src, tgt, identifier):
    map = dict()
    for i in range(len(tgt)):
        for j in tgt[i]:
            val = j.item()
            if val == identifier:
                if i not in map:
                    map[i] = 0
                else:
                    map[i] += (1/len(tgt[i]))
    sorted_items = sorted(map.items(), key=lambda x: x[1])
    result_keys = [key for key, value in sorted_items[:10]]

    #src, targ = create_balanced_dataset(train_seqs, train_target_seqs, 10, 200)
    #plot_tensor_histogram(targ)
    src = [src[x] for x in result_keys]
    tgt = [tgt[x] for x in result_keys]

    temp_x = []
    temp_y = []
    for i in range(len(src)):
        if len(src[i]) < 2000:
            temp_x.append(src[i])
            temp_y.append(tgt[i])
    src = temp_x
    tgt = temp_y

    return src, tgt


def plot_tensor_histogram(tensor_list):
    flattened = [tensor.flatten().numpy() for tensor in tensor_list]
    all_values = np.concatenate(flattened)
    unique_values, counts = np.unique(all_values, return_counts=True)

    # Plot histogram
    plt.hist(flattened, bins=len(unique_values))
    plt.xlabel('Unique Values')
    plt.ylabel('Occurrences')
    plt.title('Histogram of Unique Values')
    plt.show()


if __name__ == '__main__':

    runner = Runner()
    runner.train_model()

    #runner.split_dataset(java_dataset)
    #runner.generate_cv_folds(java_dataset)
    '''
    # cli
    SEED = 11
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_input, train_target, validation_input, validation_target = load_benchmark_data(
        'data/fold_0_train.json',
        'data/fold_0_validation.json'
    )

    train_input = train_input[:2000]
    train_target = train_target[:2000]
    input = train_input
    target = train_target
    tmp_train = list(zip(input, target))
    random.shuffle(tmp_train)
    input, target = zip(*tmp_train)


    X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.33, random_state=SEED)

    train_input, validation_input, train_target, validation_target = train_test_split(X_train, y_train, test_size=0.10, random_state=SEED)
    test_input = X_test
    test_target = y_test

    # plot occurrences
    #plot_token_occurrences(train_input, "train")
    #plot_token_occurrences(test_input, "valid")

    corpus, token2index, index2token = build_vocab(train_input, train_target, validation_input, validation_target)
    # vectorize
    train_seqs = vectorize(token2index, train_input)
    train_target_seqs = vectorize(token2index, train_target)
    validation_seqs = vectorize(token2index, validation_input)
    validation_target_seqs = vectorize(token2index, validation_target)
    test_seqs = vectorize(token2index, validation_input)
    test_target_seqs = vectorize(token2index, validation_target)
    # to tensor
    train_seqs = to_tensor(train_seqs, device)
    train_target_seqs = to_tensor(train_target_seqs, device)
    validation_seqs = to_tensor(validation_seqs, device)
    validation_target_seqs = to_tensor(validation_target_seqs, device)
    test_seqs = to_tensor(test_seqs, device)
    test_target_seqs = to_tensor(test_target_seqs, device)

    # training cli

    EPOCHS = 20
    VOCAB_SIZE = len(token2index)
    EMBED_DIM = [128]
    #LIN1_SIZE = 32 # 32
    HID_DIM = [64, 128] # 64
    #LATENT_DIM = 64 #2560 # 16
    N_LAYERS = [1, 2] # 1
    BIDIRECTIONAL = [True]

    #window_size = 512
    #stride = 512

    #train_seqs, train_target_seqs, validation_seqs, validation_target_seqs = max_pad_tensors(train_seqs, train_target_seqs, validation_seqs, validation_target_seqs)
    train_seqs, train_target_seqs = pad_tensors_to_max_length(train_seqs, train_target_seqs)
    validation_seqs, validation_target_seqs = pad_tensors_to_max_length(validation_seqs, validation_target_seqs)

    #source_windowed, target_windowed, padded_target = prepare_windows(train_seqs, train_target_seqs, window_size, stride, device)
    #validation_source_windowed, validation_target_windowed, padded_validation_target = prepare_windows(validation_seqs, validation_target_seqs, window_size, stride, device)

    for e in EMBED_DIM:
        for h in HID_DIM:
            for n in N_LAYERS:
                for b in BIDIRECTIONAL:
                    loss_function = nn.CrossEntropyLoss(ignore_index=Token.PAD)
                    # TODO: Change back model import!
                    #model = DenoisingAE(VOCAB_SIZE, EMBED_DIM, LIN1_SIZE, HID_DIM, LATENT_DIM, N_LAYERS, BIDIRECTIONAL, device)
                    model = LSTMDenoiser(VOCAB_SIZE, e, h, n, b)
                    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                    #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                           mode='max',
                                                                           factor=0.1,
                                                                           patience=10,
                                                                           cooldown=10,
                                                                           threshold=0.001,
                                                                           threshold_mode='abs',
                                                                           verbose=True)

                    print(f"{datetime.now()} - Model Config: embedding_dim={e}, hidden_dim={h}, n_layers={n}, bidirectional={b}")
                    # training
                    epoch_train_loss = []
                    epoch_valid_loss = []
                    epoch_train_acc = []
                    epoch_valid_acc = []

                    for epoch in range(EPOCHS):
                        train_loss, train_acc = train_one_epoch(
                            train_seqs,
                            train_target_seqs,
                            model,
                            loss_function,
                            optimizer
                        )
                        epoch_train_loss.append(train_loss)
                        epoch_train_acc.append(train_acc)
                        valid_loss, valid_acc = evaluate_one_epoch(
                            validation_seqs,
                            validation_target_seqs,
                            model,
                            loss_function,
                            scheduler
                        )
                        epoch_valid_loss.append(valid_loss)
                        epoch_valid_acc.append(valid_acc)

                        #predict(model, validation_seqs[0], validation_target_seqs[0], index2token)
                        #predict(model, validation_seqs[-1], validation_target_seqs[-1], index2token)

                        # predict_with_sliding_window(model, validation_source_windowed[1], validation_target_windowed[1], index2token)
                        # predict_with_sliding_window(model, validation_source_windowed[2], validation_target_windowed[2], index2token)

                        #print(
                        #    f'{datetime.now()} - Epoch {epoch + 1}: train_acc = {epoch_train_acc[epoch]}, train_loss = {epoch_train_loss[epoch]}, '
                        #    f'valid_acc = {epoch_valid_acc[epoch]}, valid_loss = {epoch_valid_loss[epoch]}'
                        #)
                    test_loss, test_acc = evaluate_one_epoch(
                        validation_seqs,
                        validation_target_seqs,
                        model,
                        loss_function,
                        scheduler
                    )
                    print(f'{datetime.now()} - train_acc = {epoch_train_acc}')
                    print(f'{datetime.now()} - train_loss = {epoch_train_loss}')
                    print(f'{datetime.now()} - valid_acc = {epoch_valid_acc}')
                    print(f'{datetime.now()} - valid_loss = {epoch_valid_loss}')
                    print(f'{datetime.now()} - Final Evaluation: test_acc = {test_acc}, test_loss = {test_loss}')
                    print('==============================================================================================================')
    '''