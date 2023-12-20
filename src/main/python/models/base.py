import torch
import random
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, fc1_size, hidden_size,
                 num_layers, latent_size, bidirectional=False, batch_size=1):
        """
        Initializes the Encoder module.

        Args:
            vocab_size (int): The size of the vocabulary which is the input size.
            embedding_size (int): The size of the embedding vectors.
            fc1_size (int): The size of the first fully connected layer before the LSTM.
            hidden_size (int): The size of the hidden state in the LSTM.
            num_layers (int): The number of layers in the LSTM.
            latent_size (int): The size of the latent space which is the output size.
            bidirectional (bool, optional): Whether the LSTM is bidirectional. Defaults to False.
            batch_size (int, optional): The batch size. Defaults to 1.
        """
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.fc1_size = fc1_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.fc1 = nn.Linear(self.embedding_size, self.fc1_size)
        self.LSTM = nn.LSTM(self.fc1_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional)
        self.fc2 = nn.Linear(self.hidden_size, self.latent_size)

    def forward(self, x):
        # x: [seq_len] -> [seq_len, 1]
        x = x.unsqueeze(1)
        embedding = self.embedding(x)
        fc1_out = self.fc1(embedding)
        fc1_out = nn.functional.relu(fc1_out)
        _, (h_s, _) = self.LSTM(fc1_out)
        h_s_last = h_s.view(self.num_layers, self.directions, self.batch_size, self.hidden_size)[-1]
        if self.bidirectional:
            h_s_last = torch.cat((h_s_last[0], h_s_last[1]))
        else:
            h_s_last = h_s_last.squeeze(0)
        fc2_out = self.fc2(h_s_last)
        fc2_out = nn.functional.relu(fc2_out)
        enc_hidden = torch.cat([fc2_out] * self.num_layers)
        return enc_hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, vocab_size, device, bidirectional=False):
        """
        Initializes the Decoder module.

        Args:
            input_size (int): The input of the decoder which is equal to batch_size since we feed in one token at a time
            hidden_size (int): The size of the hidden state in the LSTM, must match the size of the latent space of the encoder.
            num_layers (int): The number of layers in the LSTM.
            vocab_size (int): The size of the vocabulary which is the output size of the decoder.
            device (str): The device (e.g., 'cuda' or 'cpu') on which the model will be run.
            bidirectional (bool, optional): Whether the LSTM is bidirectional. Defaults to False.
        """
        super(Decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.fc_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.fc_size, self.vocab_size)

    def forward(self, y, context_vector, tfr=0.0):
        y_len = y.shape[0]
        # set teacher forcing for whole sequence
        use_tf = random.random() < tfr
        # prepare output tensor
        outputs = torch.zeros(y_len, self.vocab_size, dtype=torch.float32).to(self.device)
        # start with <SOS> token
        current_token = y[0]
        h_s = context_vector
        h_c = context_vector
        for i in range(1, y_len):
            predictions, (h_s, h_c) = self.__forward_iteration(current_token, h_s, h_c)
            outputs[i] = predictions
            best_guess = predictions.argmax(0)
            # either pass the ground truth or the earlier predicted token
            current_token = y[i] if use_tf else best_guess
        return outputs if self.training else torch.nn.functional.log_softmax(outputs, dim=1)

    def __forward_iteration(self, x, h_s_in, h_c_in):
        x = x.view(-1, 1).to(torch.float32)
        output, (h_s, h_c) = self.LSTM(x, (h_s_in, h_c_in))
        predictions = self.fc(output)
        predictions = nn.functional.relu(predictions)
        predictions = predictions.squeeze(0)
        return predictions, (h_s, h_c)


class DenoisingAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, fc1_size, hidden_size, latent_size, num_layers, bidirectional, device,
                 batch_size=1):
        """
        Initializes the custom class.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding vectors.
            fc1_size (int): The size of the first fully connected layer in the encoder.
            hidden_size (int): The size of the hidden state in the encoder.
            latent_size (int): The size of the latent space.
            num_layers (int): The number of LSTM layers for both the encoder and decoder
            bidirectional (bool): Whether the model is bidirectional.
            device (str): The device (e.g., 'cuda' or 'cpu') on which the model will be run.
            batch_size (int, optional): The batch size. Defaults to 1.
        """
        super(DenoisingAE, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_size, fc1_size, hidden_size, num_layers, latent_size, bidirectional)
        self.decoder = Decoder(batch_size, latent_size, num_layers, vocab_size, device, bidirectional)

    def forward(self, x, y, tfr=0.0):
        context_vector = self.encoder(x)
        return self.decoder(y, context_vector, tfr)
