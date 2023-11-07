import pandas as pd
from sklearn.model_selection import train_test_split
import fire
import pandas as pd
from torch import Tensor
from torch.optim import SGD, AdamW, RMSprop
import torch.nn as nn
import numpy as np
import torch
from typing import Optional, Tuple
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from metrics import write_all_metrics
from model import RNNModel, LSTMModel, GRUModel, get_available_device, TimeSeriesModel
from enum import Enum


# https://jaketae.github.io/study/pytorch-rnn/
# https://www.kaggle.com/code/pablocastilla/predict-stock-prices-with-lstm
# https://wandb.ai/wandb_fc/wb-tutorials/reports/Tutorial-Recurrent-Neural-Networks
# https://www.kaggle.com/code/namanmanchanda/rnn-in-pytorch

torch.manual_seed(42)


def convert_to_univariate(
    dataset, start_index: int, end_index: Optional[int], seq_len: int, pred_len: int
):
    """
    returns tensors of shape (samples, features)
    we return a tuple of tensors because we want to predict the next value for each
    during training, we iterate through each sample and predict the next value
    """
    data = []
    labels = []

    start_index = start_index + seq_len
    if end_index is None:
        end_index = len(dataset) - pred_len

    for i in range(start_index, end_index):
        data_indices = range(i - seq_len, i)
        label_indices = range(i, i + pred_len)
        data.append(dataset[data_indices])
        labels.append(dataset[label_indices])

    data = np.array(data)
    labels = np.array(labels)
    return torch.tensor(data), torch.tensor(labels)


class DatasetChoice(Enum):
    # https://www.kaggle.com/datasets/dgawlik/nyse/?select=prices.csv
    YAHOO = "yahoo"
    # https://github.com/VivekPa/AIAlpha/blob/master/sample_data/raw_data/price_vol.csv
    PRICE_VOL = "price_vol"


def process_yahoo(dataset: pd.DataFrame):
    """
    Returns the dataset as a series of values normalized from 0-1
    """
    yahoo = dataset[dataset["symbol"] == "YHOO"]
    yahoo_stock_prices = yahoo["close"].values.astype("float32")
    yahoo_stock_prices = yahoo_stock_prices.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    yahoo_stock_prices = scaler.fit_transform(yahoo_stock_prices)
    return yahoo_stock_prices


def process_price_vol(dataset: pd.DataFrame):
    """
    Returns the dataset as a series of values normalized from 0-1
    """
    prices = dataset["Price"].values.astype("float32")
    # add dimension
    prices = prices.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_prices = scaler.fit_transform(prices)
    return stock_prices


dataset_source_files = {
    DatasetChoice.YAHOO: "prices_sp500.csv",
    DatasetChoice.PRICE_VOL: "price_vol.csv",
}
dataset_proc_fns = {
    DatasetChoice.YAHOO: process_yahoo,
    DatasetChoice.PRICE_VOL: process_price_vol,
}


def model_step(
    model: TimeSeriesModel,
    data: Tensor,
    labels: Tensor,
    device: torch.device,
    criterion: nn.Module,
    batch_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    model.reset_hidden(batch_size)
    data = data.to(device)
    labels = labels.squeeze(-1).to(device)
    output = model(data)
    loss = criterion(output, labels)
    return loss, output, labels


def main(dataset: str = "price_vol"):
    dataset_choice = DatasetChoice(dataset)

    prices_dataset = pd.read_csv(f"data/{dataset_source_files[dataset_choice]}")
    processed_dataset = dataset_proc_fns[dataset_choice](prices_dataset)
    train_data, test_data = train_test_split(
        processed_dataset, test_size=0.2, shuffle=False
    )

    # hyperparams
    hidden_size: int = 64
    learning_rate: float = 1e-3
    # how many samples to use to predict the next sample
    sequence_length: int = 10
    prediction_length: int = 1
    n_epochs: int = 10
    batch_size: int = 512

    device = get_available_device()
    print(f"Using device: {device}")

    X_train, Y_train = convert_to_univariate(
        train_data, 0, None, sequence_length, prediction_length
    )

    X_test, Y_test = convert_to_univariate(
        test_data, 0, None, sequence_length, prediction_length
    )

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    model = GRUModel(1, hidden_size, prediction_length, device).to(device)
    model.reset_hidden(batch_size)

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    global_step = 0

    for i in range(n_epochs):
        for j, (data, labels) in enumerate(test_loader):
            with torch.no_grad():
                loss, output, labels = model_step(model, data, labels, device, criterion, batch_size)
                write_all_metrics(output, labels, writer, i)
                writer.add_scalar("test/loss", loss.item(), global_step)
                global_step += 1

                print(
                    f"Test: epoch {i:03.0f}, batch {j:03.0f}, loss {loss.item():10.8f}"
                )

        for j, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            loss, output, labels = model_step(model, data, labels, device, criterion, batch_size)
            loss.backward()
            optimizer.step()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            print(f"Train: epoch {i:03.0f}, batch {j:03.0f}, loss {loss.item():10.8f}")


if __name__ == "__main__":
    fire.Fire(main)
