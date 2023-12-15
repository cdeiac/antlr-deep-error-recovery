import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers, latent_dim, bidirectional=False, batch_size=1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(self.input_size, self.embedding_dim, dtype=torch.float32)
        self.LSTM = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        embedding = self.embedding(x)
        _, (h_s, _) = self.LSTM(embedding)
        h_s_last = h_s.view(self.num_layers, self.directions, self.batch_size, self.hidden_size)[-1]

        if self.bidirectional:
            direction_1, direction_2 = h_s_last[0], h_s_last[1]
            direction_full = torch.cat((direction_1, direction_2))
        else:
            direction_full = h_s_last.squeeze(0)

        fc_output = self.fc(direction_full)
        fc_output = nn.functional.relu(fc_output)
        return torch.cat([fc_output] * self.num_layers)




class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, output_dim, bidirectional=False):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.fc_size = self.hidden_dim*2 if self.bidirectional else self.hidden_dim

        self.LSTM = nn.LSTM(self.latent_dim, self.hidden_dim, self.num_layers, bidirectional=self.bidirectional, dtype=torch.float32)
        self.fc = nn.Linear(self.fc_size, self.output_dim, dtype=torch.float32)

    def forward(self, x, h_s_enc, h_c_enc):
        x = x.reshape(-1, 1).to(torch.float32)
        output, (h_s, h_c) = self.LSTM(x, (h_s_enc, h_c_enc))
        predictions = self.fc(output)
        predictions = predictions.squeeze(0)
        return predictions, (h_s, h_c)


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, device, bidirectional=False, batch_size=1):
        super(DenoisingAutoEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, latent_dim, bidirectional)
        self.decoder = Decoder(batch_size, latent_dim, num_layers, vocab_size, bidirectional)

    def forward(self, x, y, tfr=0.0):
        #batch_size = 1 if x.dim()==1 else x.shape[1]
        y_len = y.shape[0]
        outputs = torch.zeros(y_len, self.vocab_size, dtype=torch.float32).to(self.device)
        h_s_initial = self.encoder(x)
        x = y[0] # Start with <SOS> token
        h_s = h_s_initial
        h_c = h_s_initial
        for i in range(1, y_len):
            # decoder
            predictions, (h_s, h_c) = self.decoder(x, h_s, h_c)
            outputs[i] = predictions
            best_guess = predictions.argmax(0)
            # Either pass the ground truth or use the earlier predicted token
            x = y[i] if random.random() < tfr else best_guess
        return outputs if self.training else torch.nn.functional.log_softmax(outputs, dim=1)
