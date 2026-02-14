import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.gru(x)
        x = self.fc(x)
        return x