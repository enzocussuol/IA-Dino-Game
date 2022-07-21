import torch.nn as nn
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class NN(nn.Module):
    def __init__(self, num_imputs, num_outputs):
        super(NN, self).__init__()
        num_internal_neurons = num_imputs*2 + 1
        self.layers = nn.Sequential(
            nn.Linear(num_imputs, num_internal_neurons),
            nn.ReLU(),
            nn.Linear(num_internal_neurons, num_outputs),
            nn.ReLU()
        )
        
    def forward(self, x):
        result_tensor = self.layers(x)

        if result_tensor[0] > 0:
            return "K_UP"
        elif result_tensor[1] > 0:
            return "K_DOWN"
        return "K_NO"