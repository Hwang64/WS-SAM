
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIDN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc_cls = nn.Linear(input_dim, num_classes)
        self.fc_det = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        score_cls = F.softmax(self.fc_cls(features), dim=-1)
        score_det = F.softmax(self.fc_det(features), dim=-1)
        score = score_cls * score_det
        return score.sum(dim=1)
