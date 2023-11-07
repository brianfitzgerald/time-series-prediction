import pandas as pd
from torch import Tensor
import torch
from torch.utils.tensorboard.writer import SummaryWriter


def get_r_squared(preds: Tensor, labels: Tensor):
    """
    Calculate the R^2 value for the given predictions and labels.
    """
    sum_squared_regression = ((labels - preds) ** 2).sum()
    sum_of_squares = ((labels - labels.mean()) ** 2).sum()
    return 1 - (sum_squared_regression / sum_of_squares)


def get_mae(preds: Tensor, labels: Tensor):
    """
    Calculate the mean absolute error for the given predictions and labels.
    """
    return (labels - preds).abs().mean()


def get_mape(preds: Tensor, labels: Tensor):
    """
    Calculate the mean absolute percentage error for the given predictions and labels.
    """

    percentage_error = ((labels - preds) / labels).abs()

    return percentage_error.mean()


def get_smape(preds: Tensor, labels: Tensor):
    """
    Calculate the symmetric mean absolute percentage error for the given predictions and labels.
    """

    abs_residuals = (labels - preds).abs()
    denominator = (labels.abs() + preds.abs()) / 2

    return (100 * abs_residuals / denominator).mean()

def get_rmse(preds: Tensor, labels: Tensor):
    """
    Calculate the root mean squared error for the given predictions and labels.
    """
    return torch.sqrt(((labels - preds) ** 2).mean())


def write_all_metrics(
    preds: Tensor,
    labels: Tensor,
    writer: SummaryWriter,
    epoch: int,
):
    """
    Write all metrics to the SummaryWriter provided.
    """
    

    writer.add_scalar("val/r_squared", get_r_squared(preds, labels), epoch)
    writer.add_scalar("val/mae", get_mae(preds, labels), epoch)
    writer.add_scalar("val/mape", get_mape(preds, labels), epoch)
    writer.add_scalar("val/smape", get_smape(preds, labels), epoch)
    writer.add_scalar("val/rmse", get_rmse(preds, labels), epoch)
