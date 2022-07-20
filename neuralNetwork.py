import torch.nn as nn

class NN(nn.module):
    def __init__(self, num_imputs, num_outputs):
        super(NN, self).__init__()
        num_internal_neurons = num_imputs*2 + 1
        self.layers = nn.Sequential(
            nn.Linear(num_imputs, num_internal_neurons),
            nn.ReLU(),
            nn.Linear(num_internal_neurons, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, 3)
        )
        
    def forward(self, x):
        return self.layers(x)