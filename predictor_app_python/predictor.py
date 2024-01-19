'''
    @author:
        - Leonardo Lo Schiavo
    @affiliation:
        - IMDEA Networks institute
'''
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, nonlin=F.relu, norm_in=False):

        super(Predictor, self).__init__()

        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_size, affine=False)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.nonlin = nonlin


    def forward(self, X):

        inp = self.in_fn(X)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out