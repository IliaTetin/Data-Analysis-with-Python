import numpy as np
import json
import re
import torch
import torch.nn as nn

def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10
    # your code here
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )
    # return model instance (None is just a placeholder)

    return model

def count_parameters(model):
    # your code here
    count = 0
    for param in model.parameters():
        count += param.numel()

    # верните количество параметров модели model
    return count
