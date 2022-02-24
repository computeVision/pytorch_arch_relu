import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_input, num_output):
        super(Network, self).__init__()
        self.relu_1 = ReLULayer(num_input, 10)
        self.relu_2 = ReLULayer(10, 10)
        self.output = nn.Linear(10, num_output)

    def forward(self, x):
        x = self.relu_1(x)
        x = self.relu_2(x)
        x = self.output(x)
        return x

        