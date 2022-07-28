from unittest import result
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import argmax

class NN(nn.Module):
    def __init__(self, num_imputs, num_outputs):
        super(NN, self).__init__()
        num_internal_neurons = int((num_imputs*2)/3 + num_outputs)
        self.layers = nn.Sequential(
            nn.Linear(num_imputs, num_internal_neurons),
            nn.ReLU(),
            nn.Linear(num_internal_neurons, num_outputs),
            nn.ReLU()
        )
        
    def forward(self, x):
        result_tensor = self.layers(x)
        result = argmax(result_tensor).item()

        if result == 0:
            return "K_UP"
        elif result == 1:
            return "K_DOWN"
        return "K_NO"