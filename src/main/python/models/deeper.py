import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_size, num_of_lstms, lin1_size, lin2_size, bidirectional=False):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_size = lstm_size
        self.num_of_lstms = num_of_lstms
        self.bidirectional = bidirectional
        self.lin1_size = lin1_size
        self.lin2_size = lin2_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, dtype=torch.float32)
        self.LSTM = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_of_lstms, bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(self.lstm_size, self.lin1_size)
        self.fc2 = nn.Linear(self.lin1_size, self.lin2_size)

    def forward(self, x):
        embedding = self.embedding(x)
        _, (h_s, _) = self.LSTM(embedding)
        h_s_last = h_s[-1]

        fc1_output = self.fc1(h_s_last)
        fc1_output = nn.functional.relu(fc1_output)
        fc2_output = self.fc2(fc1_output)
        fc2_output = nn.functional.relu(fc2_output)

        return torch.cat([fc2_output.unsqueeze(0)] * self.num_of_lstms)


class Decoder(nn.Module):
    def __init__(self, input_size, lstm_size, num_of_lstms, vocab_size, bidirectional=False):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.num_of_lstms = num_of_lstms
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.lstm_size = lstm_size
        self.fc_size = self.lstm_size * 2 if self.bidirectional else self.lstm_size

        self.LSTM = nn.LSTM(self.input_size, self.lstm_size, self.num_of_lstms, bidirectional=self.bidirectional,
                            dtype=torch.float32)
        self.fc = nn.Linear(self.fc_size, self.vocab_size, dtype=torch.float32)

    def forward(self, x, h_s_enc, h_c_enc):
        x = x.reshape(-1, 1).to(torch.float32)
        output, (h_s, h_c) = self.LSTM(x, (h_s_enc, h_c_enc))
        predictions = self.fc(output)
        predictions = predictions.squeeze(0)
        return predictions, (h_s, h_c)


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_size, lin1_size, lin2_size, num_of_lstms, device, batch_size=1,
                 bidirectional=False):
        super(DenoisingAutoEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.batch_size = batch_size
        self.encoder = Encoder(vocab_size, embedding_dim, lstm_size, num_of_lstms, lin1_size, lin2_size, bidirectional)
        self.decoder = Decoder(batch_size, lin2_size, num_of_lstms, vocab_size, bidirectional)

    def forward(self, x, y, tfr=0.0):
        #batch_size = 1 if x.dim() == 1 else x.shape[1]
        y_len = y.shape[0]
        outputs = torch.zeros(y_len, self.vocab_size, dtype=torch.float32).to(self.device)
        h_s_initial = self.encoder(x)
        x = y[0]  # Start with <SOS> token
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
