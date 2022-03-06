import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

VOCAB = {'A','B','C','D','E'}



class IMDBModel(nn.Module):
    def __init__(self, max_len):
        super(IMDBModel, self).__init__()
        self.embedding = nn.Embedding(5,300,) # [N,300]
        self.fc = nn.Linear(max_len * 300, 10)  # [max_len*300,10]

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size,max_len,300]
        embed = embed.view(x.size(0), -1)
        out = self.fc(embed)
        return F.log_softmax(out, dim=-1)
