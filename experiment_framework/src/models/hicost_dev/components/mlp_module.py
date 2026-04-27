import torch 
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(dim, 250)
        self.fc_2 = nn.Linear(250, 50)
        self.fc_3 = nn.Linear(50, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class RestartMLP(nn.Module):
    def __init__(self, dim, drop=0.2):
        super().__init__()
        self.fc_1 = nn.Linear(dim, 80)
        self.fc_2 = nn.Linear(80, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.fc_2(x)
        x = torch.sigmoid(x)
        return x