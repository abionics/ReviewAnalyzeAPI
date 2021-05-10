import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


class CustomLSTM(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        # initialize the hidden state (see code below)
        self.hidden_dim = self.init_hidden()

    def init_hidden(self):
        """At the start of training, we need to initialize a hidden state
        there will be none because the hidden state is formed based on perviously seen data
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


class ReviewsDataset(Dataset):
    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.x[idx][0].astype(np.int32))
        return tensor, self.y[idx], self.x[idx][1]


def validation_metrics(model: torch.nn.Module, dataloader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in dataloader:
        print(x, type(x))
        print(l, type(l))
        x = x.long().to(device)
        y = y.long().to(device)
        print(x, type(x))
        print(l, type(l))
        y_hat = model(x, l)
        print(y_hat)
        exit(2)
        loss = functional.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1].cpu()
        correct += (pred == y.cpu()).float().sum().cpu()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.cpu().unsqueeze(-1))) * y.shape[0]
    return sum_loss / total, correct / total, sum_rmse / total


def load(filename: str = 'data.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)


print('CUDA:', torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()

model = CustomLSTM(2002, 100, 100)
model.to(device)

model.load_state_dict(torch.load('ai/model.pt', map_location=torch.device('cpu')))
model.eval()

test = load('ai/pickles/test.pkl')

indexes = []
for index, row in test.iterrows():
    if row['encoded'][0][0] == 0:
        indexes.append(index)
test.drop(test.index[indexes], inplace=True)

zero_numbering = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
test['label'] = test['label'].apply(lambda x: zero_numbering[x])

# Create dataloader for test dataset
x_test = list(test['encoded'])
y_test = list(test['label'])
test_dataset = ReviewsDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=2000)

# Evaluate model on test dataset
print('xxx')
test_loss, test_acc, test_rmse = validation_metrics(model, test_dataloader)
print('test loss %.3f, test accuracy %.3f, and test rmse %.3f' % (test_loss, test_acc, test_rmse))
