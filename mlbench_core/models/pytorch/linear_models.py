import torch
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class LogisticRegression(torch.nn.Module):
    """
    Logistic regression implementation

    Args:
        n_features (int): Number of features
    """

    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()

        self.linear = torch.nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


class LinearRegression(torch.nn.Module):
    """
    Ridge regression implementation

    Args:
        n_features (int): Number of features
    """

    def __init__(self, n_features):
        super(RidgeRegression, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        return self.linear(x)

