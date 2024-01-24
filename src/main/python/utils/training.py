import torch
import matplotlib.pyplot as plt

from dataset.token import Token
from utils.evaluation import compute_accuracy, aggregated_accuracy
from utils.training_session import TrainingSession


def generate_windows(x, y, device, target_dim=0, window_size=64, step_size=64, pad_value=Token.PAD):
    # select the longer sequence + add one additional step
    longest_sequence = max(x.shape[0], y.shape[0])
    x_adjustment = (longest_sequence - x.shape[0])
    y_adjustment = (longest_sequence - y.shape[0])
    pad_len_all, pad_len_x, pad_len_y = 0, 0, 0


    # (-1 to account for SOS to be inserted)
    # TODO: change formula for when stride is not equals to window size
    pad_len_all = ((window_size-1) - longest_sequence % (window_size-1))
    # pad sequence to match window size
    pad_len_x = (pad_len_all + x_adjustment)
    pad_len_y = (pad_len_all + y_adjustment)
    x = torch.nn.functional.pad(input=x, pad=(0, pad_len_x), mode='constant', value=pad_value)
    y = torch.nn.functional.pad(input=y, pad=(0, pad_len_y), mode='constant', value=pad_value)
    # prepend SOS token for each window
    #x = torch.cat((torch.tensor([Token.SOS]).to(device), x), dim=0)
    #y = torch.cat((torch.tensor([Token.SOS]).to(device), y), dim=0)
    #window_size += 1
    # generate sliding windows
    x_windowed = x.unfold(target_dim, window_size-1, step_size-1)
    y_windowed = y.unfold(target_dim, window_size-1, step_size-1)
    x_windowed_expanded = torch.cat((torch.full((x_windowed.size(0), 1), Token.SOS), x_windowed), dim=1)
    y_windowed_expanded = torch.cat((torch.full((y_windowed.size(0), 1), Token.SOS), y_windowed), dim=1)
    return x_windowed_expanded, y_windowed_expanded


def generate_windows_for_dataset(source, target, device, target_dim=0, window_size=64, step_size=64, pad_value=Token.PAD):
    source_windowed = []
    target_windowed = []
    for i in range(len(source)):
        src_w, trg_w = generate_windows(source[i][1:], target[i][1:], device)
        source_windowed.append(src_w.to(device))
        target_windowed.append(trg_w.to(device))
    return source_windowed, target_windowed


def decode(tensor, idx2token):
    result = ''
    for id in tensor:
        result += f' {idx2token[id]}'
    return result


def train_one_epoch(
        sources: [torch.Tensor],
        targets: [torch.Tensor],
        training_session: TrainingSession
):
    model = training_session.model
    optimiser = training_session.optimiser
    loss_function = training_session.loss_function
    n = len(sources)
    running_loss = 0.0
    running_acc = 0.0
    model.train()
    for source, target in zip(sources, targets):
        optimiser.zero_grad()
        # we want the model to predict the next token, therefore: skip source EOS and target SOS
        source = source[:-1]
        target = target[1:]
        # produce output
        output = model(source)
        # compute loss and accuracy
        loss = loss_function(output, target)
        running_loss += loss.item()
        prediction = torch.argmax(output, dim=1)
        running_acc += compute_accuracy(prediction, target)
        # backpropagation
        loss.backward()
        optimiser.step()
    return running_loss/n, running_acc/n


def evaluate_one_epoch(
        sources: [torch.Tensor],
        targets: [torch.Tensor],
        training_session: TrainingSession
):
    model = training_session.model
    loss_function = training_session.loss_function
    scheduler = training_session.scheduler
    model.eval()
    with torch.no_grad():
        n = len(sources)
        running_loss = 0.0
        running_acc = 0.0
        for source, target in zip(sources, targets):
            # we want the model to predict the next token, therefore: skip source EOS and target SOS
            source = source[:-1]
            target = target[1:]
            # produce output
            output = model(source)
            # compute loss and accuracy
            loss = loss_function(output, target)
            running_loss += loss.item()
            prediction = torch.argmax(output, dim=1)
            running_acc += compute_accuracy(prediction, target)
        # trigger scheduler on accuracy
        epoch_val_loss = running_loss/n
        epoch_val_acc = running_acc/n
        scheduler.step(epoch_val_acc)
        return epoch_val_loss, epoch_val_acc


def predict(
        sources: [torch.Tensor],
        targets: [torch.Tensor],
        training_session: TrainingSession
):
    model = training_session.model
    model.eval()
    with torch.no_grad():
        n = len(sources)
        running_acc = 0.0
        for source, target in zip(sources, targets):
            # we want the model to predict the next token, therefore: skip source EOS and target SOS
            source = source[:-1]
            target = target[1:]
            # produce output
            output = model(source)
            # compute accuracy
            prediction = torch.argmax(output, dim=1)
            running_acc += compute_accuracy(prediction, target)
        return running_acc / n

'''
def training(config: Config, model, loss_function, optimizer, scheduler, source_train_data,
             target_train_data, source_valid_data, target_valid_data):
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for epoch in range(config.epochs):
        source_train_windowed, target_train_windowed = generate_windows_for_dataset(
            source_train_data,
            target_train_data,
            config.device
        )
        source_valid_windowed, target_valid_windowed = generate_windows_for_dataset(
            target_valid_data,
            source_valid_data,
            config.device
        )

        train_loss, train_acc = train_one_epoch_with_windowing(source_train_windowed,
                                                               target_train_windowed,
                                                               model,
                                                               loss_function,
                                                               optimizer)
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        # for j in range(len(test_batch)):
        valid_loss, valid_acc = evaluate_one_epoch_with_windowing(source_valid_windowed,
                                                                  target_valid_windowed,
                                                                  model,
                                                                  loss_function,
                                                                  scheduler)
        epoch_valid_loss.append(valid_loss)
        epoch_valid_acc.append(valid_acc)
        print(
            f'Epoch {epoch + 1}: train_acc = {epoch_train_acc[epoch]}, train_loss = {epoch_train_loss[epoch]}, '
            f'valid_acc = {epoch_valid_acc[epoch]}, valid_loss = {epoch_valid_loss[epoch]}'
        )


def train_one_epoch(source, target, model, loss_function, optimizer):
    n = len(source)
    train_acc = 0.0
    train_loss = 0.0
    model.train()
    for i in range(n):
        optimizer.zero_grad()
        output = model(source[i], target[i], 0.5) # teacher forcing
        loss = loss_function(output, target[i][1:])
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters())
        optimizer.step()
        train_loss += loss.item()
        _, topi = output.topk(1)
        predicted = topi.squeeze()
        train_acc += compute_accuracy(predicted, target[i][1:])
    return train_loss/n, train_acc/n


def train_one_epoch_with_windowing(source_windowed, target_windowed, targets, model, loss_function, optimizer, stride):
    n = len(source_windowed)
    train_acc = 0.0
    train_loss = 0.0
    model.train()
    for j in range(n):
        if j % 100 == 0:
            print(f'{j}/{n}')
        predicted_windows = []
        src, trg = source_windowed[j], target_windowed[j]
        target = targets[j]
        n_w = len(src)
        #running_train_acc = 0.0
        running_train_loss = 0.0
        for i, (x,y) in enumerate(zip(src, trg)):
            is_last_window = i == n_w-1
            #print(f'    {i}/{n_w}')
            optimizer.zero_grad()
            #y = y[1:]
            #if not is_last_window:
            #    y=y[0]
            output = model(x, y, stride, 0.5) # teacher forcing
            #output = output[1:].reshape(-1, output.shape[-1])
            y = y[1:]
            # TODO: if last window, loss_function(output, stride) if no additional padding was added
            loss = loss_function(output[:stride], y[:stride])
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters())
            optimizer.step()
            running_train_loss += loss.item()

            predicted = torch.argmax(output, dim=1) #torch.argmax(output, dim=0 if not is_last_window else 1)
            predicted_windows.append(predicted)
            #_, topi = output.topk(1)
            #predicted = topi.squeeze()
            #if j == 0 and i < 2:
            #    print(f"Targ: {y.tolist()}")
            #    print(f"Pred: {predicted.tolist()}")
            #running_train_acc += compute_accuracy(predicted, y) #sum(predicted == y) / y.shape[0]).item()
        train_loss += running_train_loss/(i+1)
        #train_acc += running_train_acc/(i+1)
        train_acc += aggregated_accuracy(predicted_windows, target[1:], stride)
    return train_loss/n, train_acc/n


def evaluate_one_epoch(source, target, model, loss_function, scheduler):
    n = len(source)
    valid_acc = 0.0
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(n):
            output = model(source[i], target[i]) # no teacher forcing
            loss = loss_function(output, target[i][1:])
            valid_loss += loss.item()
            _, topi = output.topk(1)
            predicted = topi.squeeze()
            valid_acc += compute_accuracy(predicted, target[i][1:]) #sum(predicted == y) / y.shape[0]).item()
    epoch_valid_acc = valid_acc/n
    epoch_valid_loss = valid_loss/n
    scheduler.step(epoch_valid_acc)
    return epoch_valid_loss, epoch_valid_acc


def evaluate_one_epoch_with_windowing(source, target, padded_target, model, loss_function, scheduler, stride):
    n = len(source)
    valid_acc = 0.0
    valid_loss = 0.0
    model.eval()
    predicted_windows = []
    with torch.no_grad():
        for j in range(len(source)):
            src, trg = source[j], target[j]
            pad_target = padded_target[j]
            running_valid_loss = 0.0
            for i, (x,y) in enumerate(zip(src, trg)):
                output = model(x, y, stride) # no teacher forcing
                #output = output[1:].reshape(-1, output.shape[-1])
                y = y[1:]
                loss = loss_function(output[:stride], y[:stride])
                running_valid_loss += loss.item()

                predicted = torch.argmax(output, dim=1)  # torch.argmax(output, dim=0 if not is_last_window else 1)
                predicted_windows.append(predicted)
                #running_valid_acc += compute_accuracy(predicted, y)#(sum(predicted == y) / y.shape[0]).item()
            valid_loss += running_valid_loss/(i+1)
            #valid_acc += running_valid_acc/(i+1)
            valid_acc += aggregated_accuracy(predicted_windows, pad_target[1:], stride)

    epoch_valid_acc = valid_acc/n
    epoch_valid_loss = valid_loss/n
    scheduler.step(epoch_valid_acc)
    return epoch_valid_loss, epoch_valid_acc


def save_model(path, epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
               train_loss, train_acc, valid_loss, valid_acc):
    torch.save({
        'epoch': epoch+1,
        'model': model_state_dict,
        'optimizer': optimizer_state_dict,
        'scheduler': scheduler_state_dict,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'validation_loss': valid_loss,
        'validation_acc': valid_acc
    }, f'{path}/{epoch+1}.pt')


def predict(model, input, target, idx2token):
    model.eval()
    #for i, (x, y) in enumerate(zip(input, target)):
    out = model(input, target)
    _, topi = out.topk(1)
    decoded_ids = topi.squeeze()
    print(f'Targ: {decode(target[1:].tolist(), idx2token)}')
    print(f'Pred: {decode(decoded_ids.tolist(), idx2token)}')


def predict_with_sliding_window(model, input, target, idx2token):
    model.eval()
    for i, (x, y) in enumerate(zip(input, target)):
        if i > 0:
            break
        out = model(x, y)
        out = out[1:].reshape(-1, out.shape[-1])
        y = y[1:].reshape(-1)
        _, topi = out.topk(1)
        decoded_ids = topi.squeeze()
        print(f'Targ: {decode(y.tolist(), idx2token)}')
        print(f'Pred: {decode(decoded_ids.tolist(), idx2token)}')

'''

def plot(train_loss, epochs, valid_loss, title):
    plt.plot(list(range(epochs)), train_loss, label='train loss', color='blue')
    plt.plot(list(range(epochs)), valid_loss, label='validation loss', color='orange')
    plt.xlim(1, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title, fontsize = 12)
    plt.legend()
    plt.xticks(list(range(epochs)))
    plt.show()
