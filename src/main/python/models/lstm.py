import torch
from torch import nn


class LSTMDenoiser(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, bidirectional):
        super(LSTMDenoiser, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        #embedded = torch.nn.functional.relu(embedded)
        output, _ = self.lstm(embedded)
        #output = torch.nn.functional.relu(output)
        output = self.linear(output)
        return output if self.training else torch.nn.functional.log_softmax(output, dim=1)
