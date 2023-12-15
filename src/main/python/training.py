import torch
import matplotlib.pyplot as plt
from data import generate_rolling_windows


def decode(tensor, idx2token):
    result = ''
    for id in tensor:
        result += f' {idx2token[id]}'
    return result


def train_one_epoch(input_data, target_data, model, optimizer, loss_function):
    model.train()
    train_loss = 0
    for i in range(len(input_data)):
        # push tensors to device before
        input = input_data[i]
        target = target_data[i]
        # print(f'Input: {input.shape}')
        # Pass the input and target for model's forward method
        output = model(input, target, 0.5)  # teacher forcing
        # print(f'1.) Output shape: {output.shape}')
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        # print(f'2.) Output shape: {output.shape}')
        target = target[1:].reshape(-1)
        # print(f'3.) Target shape: {target.shape}')
        # Clear the accumulating gradients
        optimizer.zero_grad()
        # Calculate the loss value for every epoch
        loss = loss_function(output, target)
        # Calculate the gradients for weights & biases using back-propagation
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(input_data)


def train_one_epoch_with_sliding_window(input_data, target_data, model, optimizer, loss_function):
    model.train()
    train_loss = 0
    for n in range(len(input_data)):
        # push tensors to device before
        input, target = generate_rolling_windows(input_data[n], target_data[n])
        seq_train_loss = 0.0
        for i, (x, y) in enumerate(zip(input, target)):
            # Pass the input and target for model's forward method
            output = model(x, y, 0.5)  # teacher forcing
            output = output[1:].reshape(-1, output.shape[-1])
            y = y[1:].reshape(-1)
            # Clear the accumulating gradients
            optimizer.zero_grad()
            # Calculate the loss value for every epoch
            loss = loss_function(output, y)
            # Calculate the gradients for weights & biases using back-propagation
            loss.backward()
            optimizer.step()
            seq_train_loss += loss.item()
        train_loss += seq_train_loss / i
    return train_loss / len(input_data)


def evaluate_one_epoch(input_data, target_data, model, loss_function, scheduler, idx2token):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i in range(len(input_data)):
            # push tensors to device before
            input = input_data[i]
            target = target_data[i]
            output = model(input, target)  # no teacher forcing
            output_dim = output.shape[-1]  # get last dimension (embedding)
            output = output[1:].reshape(-1, output_dim)  # flatten output
            target = target[1:].reshape(-1)  # flatten target
            loss = loss_function(output, target)
            valid_loss += loss.item()
            # predict
            # print(f'Model Output shape: {model_output.shape}')
            #_, topi = output.topk(1)
            #decoded_ids = topi.squeeze()
            #print(f'IDs: {decoded_ids.tolist()}')
            #print(f'Prediction: {decode(decoded_ids.tolist(), idx2token)}')
    running_valid_loss = valid_loss / len(input_data)
    scheduler.step(running_valid_loss)
    return running_valid_loss


def evaluate_one_epoch_with_sliding_window(input_data, target_data, model, loss_function, scheduler, idx2token):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for n in range(len(input_data)):
            # push tensors to device before
            input, target = generate_rolling_windows(input_data[n], target_data[n])
            seq_valid_loss = 0.0
            for i, (x, y) in enumerate(zip(input, target)):
                output = model(x, y)  # no teacher forcing
                output = output[1:].reshape(-1, output.shape[-1])  # flatten output
                y = y[1:].reshape(-1)  # flatten target
                loss = loss_function(output, y)
                seq_valid_loss += loss.item()
            valid_loss += loss.item() / i
    running_valid_loss = valid_loss / len(input_data)
    scheduler.step(running_valid_loss)
    return running_valid_loss


def predict_with_sliding_window(model, input, target, idx2token):
    model.eval()
    inp, trg = generate_rolling_windows(input, target)
    for i, (x, y) in enumerate(zip(inp, trg)):
        out = model(x, y)
        out = out[1:].reshape(-1, out.shape[-1])
        y = y[1:].reshape(-1)
        _, topi = out.topk(1)
        decoded_ids = topi.squeeze()
        print(f'Target: {decode(y.tolist(), idx2token)}')
        print(f'Prediction: {decode(decoded_ids.tolist(), idx2token)}')


def plot (train_loss, epochs, valid_loss, title):
    plt.plot(list(range(epochs)), train_loss, label='train loss', color='blue')
    plt.plot(list(range(epochs)), valid_loss, label='validation loss', color='orange')
    plt.xlim(1, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title, fontsize = 12)
    plt.legend()
    plt.xticks(list(range(epochs)))
    plt.show()
