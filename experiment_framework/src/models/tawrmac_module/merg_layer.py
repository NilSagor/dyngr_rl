import torch
import torch.nn as nn

class AffinityMergeLayer(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3 * 2)
        self.fc2 = nn.Linear(dim3 * 2, dim3)
        self.fc3 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop, inplace=False)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.act(x)
        return self.fc3(x)


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, dim3)
        self.fc2 = nn.Linear(dim3, dim4)
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)