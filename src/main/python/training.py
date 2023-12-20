import torch
import matplotlib.pyplot as plt

from config import Config
from dataset import Dataset


def training(config: Config, model, loss_function, optimizer, scheduler, source_train_data,
             target_train_data, source_valid_data, target_valid_data):
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for epoch in range(config.epochs):
        source_train_windowed, target_train_windowed = Dataset.generate_windows_for_dataset(
            source_train_data,
            target_train_data,
            config.device
        )
        source_valid_windowed, target_valid_windowed = Dataset.generate_windows_for_dataset(
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
            f'Epoch {epoch + 1}: train_acc = {epoch_train_acc[epoch]}, train_loss = {epoch_train_loss[epoch]}, valid_acc = {epoch_valid_acc[epoch]}, valid_loss = {epoch_valid_loss[epoch]}')


def decode(tensor, idx2token):
    result = ''
    for id in tensor:
        result += f' {idx2token[id]}'
    return result


def train_one_epoch(source, target, model, loss_function, optimizer):
    n = len(source)
    train_acc = 0.0
    train_loss = 0.0
    model.train()
    for i in range(len(source)):
        model.zero_grad()
        optimizer.zero_grad()
        output = model(source[i], target[i], 0.5) # teacher forcing
        output = output[1:].reshape(-1, output.shape[-1])
        y = y[1:].reshape(-1)
        loss = loss_function(output, y)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters())
        optimizer.step()
        train_loss += loss.item()
        _, topi = output.topk(1)
        predicted = topi.squeeze()
        train_acc += (((predicted == y).sum()) / y.shape[0]).item()
    return train_loss/n, train_acc/n


def train_one_epoch_with_windowing(source, target, model, loss_function, optimizer):
    n = len(source)
    train_acc = 0.0
    train_loss = 0.0
    model.train()
    for i in range(len(source)):
        src, trg = source[i], target[i]
        running_train_acc = 0.0
        running_train_loss = 0.0
        for i, (x,y) in enumerate(zip(src, trg)):
            model.zero_grad()
            optimizer.zero_grad()
            output = model(x, y, 0.5) # teacher forcing
            output = output[1:].reshape(-1, output.shape[-1])
            y = y[1:].reshape(-1)
            loss = loss_function(output, y)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters())
            optimizer.step()
            running_train_loss += loss.item()
            _, topi = output.topk(1)
            predicted = topi.squeeze()
            running_train_acc += (((predicted == y).sum()) / y.shape[0]).item()
        train_loss += running_train_loss/(i+1)
        train_acc += running_train_acc/(i+1)
    return train_loss/n, train_acc/n


def evaluate_one_epoch(source, target, model, loss_function, scheduler):
    n = len(source)
    valid_acc = 0.0
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(len(source)):
            output = model(source[i], target[i]) # no teacher forcing
            output = output[1:].reshape(-1, output.shape[-1])
            y = y[1:].reshape(-1)
            loss = loss_function(output, y)
            valid_loss += loss.item()
            _, topi = output.topk(1)
            predicted = topi.squeeze()
            valid_acc += (sum(predicted == y) / y.shape[0]).item()
    epoch_valid_acc = valid_acc/n
    epoch_valid_loss = valid_loss/n
    scheduler.step(epoch_valid_acc)
    return epoch_valid_loss, epoch_valid_acc


def evaluate_one_epoch_with_windowing(source, target, model, loss_function, scheduler):
    n = len(source)
    valid_acc = 0.0
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for j in range(len(source)):
            src, trg = source[j], target[j]
            running_valid_loss = 0.0
            running_valid_acc = 0.0
            for i, (x,y) in enumerate(zip(src, trg)):
                output = model(x, y) # no teacher forcing
                output = output[1:].reshape(-1, output.shape[-1])
                y = y[1:].reshape(-1)
                loss = loss_function(output, y)
                running_valid_loss += loss.item()
                _, topi = output.topk(1)
                predicted = topi.squeeze()
                running_valid_acc += (sum(predicted == y) / y.shape[0]).item()
            valid_loss += running_valid_loss/(i+1)
            valid_acc += running_valid_acc/(i+1)

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


def predict_with_sliding_window(model, input, target, idx2token):
    model.eval()
    inp, trg = Dataset.generate_windows(input, target)
    for i, (x, y) in enumerate(zip(inp, trg)):
        out = model(x, y)
        out = out[1:].reshape(-1, out.shape[-1])
        y = y[1:].reshape(-1)
        _, topi = out.topk(1)
        decoded_ids = topi.squeeze()
        print(f'Target: {decode(y.tolist(), idx2token)}')
        print(f'Prediction: {decode(decoded_ids.tolist(), idx2token)}')


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
