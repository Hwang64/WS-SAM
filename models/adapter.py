
import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalAdapter(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.conv = nn.Conv2d(proj_dim, proj_dim, 1)
        self.fc2 = nn.Linear(proj_dim, in_dim)
        self.scale_fc = nn.Parameter(torch.tensor(0.1))

        self.down = nn.Linear(in_dim, proj_dim)
        self.up = nn.Linear(proj_dim, in_dim)
        self.scale_mlp = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, mha_out, is_conv=False):
        residual = x
        x = self.fc1(x)
        if is_conv:
            B, N, D = x.shape
            h = w = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, -1, h, w)
            x = self.conv(x)
            x = x.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        x = residual + self.scale_fc * x

        residual2 = x
        mlp_out = F.relu(self.down(x + mha_out))
        mlp_out = self.up(mlp_out)
        x = residual2 + self.scale_mlp * mlp_out
        return x
