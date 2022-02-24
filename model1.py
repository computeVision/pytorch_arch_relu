import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_input, num_output):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(num_input, 10)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(10, num_output)

    def forward(self, x):
        x = self.relu_1(self.fc1(x))
        x = self.relu_2(self.fc2(x)
        x = self.fc3(x)
        return x