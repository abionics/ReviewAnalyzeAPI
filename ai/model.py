import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from config import MODEL_FILENAME


class CustomLSTM(torch.nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        # initialize the hidden state
        self.hidden_dim = self.init_hidden(device)

    def init_hidden(self, device: torch.device):
        """At the start of training, we need to initialize a hidden state
        there will be none because the hidden state is formed based on previously seen data
        So, this function defines a hidden state with all zeroes and of a specified size"""
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return torch.zeros(1, 1, self.hidden_dim).to(device), \
               torch.zeros(1, 1, self.hidden_dim).to(device)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out

    @staticmethod
    def load_from_file(device: torch.device, capacity: int) -> 'CustomLSTM':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = CustomLSTM(capacity, 100, 100, device)
        model.to(device)

        state_dict = torch.load(MODEL_FILENAME, map_location=device)
        model.load_state_dict(state_dict)
        return model
