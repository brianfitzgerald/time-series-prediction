from typing import TypeAlias
import torch
import torch.nn as nn
from torch import Tensor


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        # num_features, hidden_size, num_layers
        # features means the no. of features per sample
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, seq: Tensor):
        h0, c0 = self.hidden
        lstm_out, self.hidden = self.lstm(seq, (h0, c0))
        pred = self.linear(lstm_out)
        return pred[:, -1]

    def reset_hidden(self, batch_size: int):
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
        )



class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        # num_features, hidden_size, num_layers
        # features means the no. of features per sample
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, seq: Tensor):
        h0, c0 = self.hidden
        lstm_out, self.hidden = self.lstm(seq, (h0, c0))
        pred = self.linear(lstm_out)
        return pred[:, -1]

    def reset_hidden(self, batch_size: int):
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
        )


class RNNModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        # num_features, hidden_size, num_layers
        # features means the length of the vector representing the input
        self.rnn = nn.RNN(
            n_features, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x: Tensor):
        rnn_out, self.hidden = self.rnn(x, self.hidden)
        pred = self.linear(rnn_out)
        # only return the last prediction
        return pred[:, -1]

    def reset_hidden(self, batch_size: int):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(
            self.device
        )


TimeSeriesModel: TypeAlias = LSTMModel | RNNModel | GRUModel


def get_available_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # type: ignore
        device = torch.device("mps")
    return device
