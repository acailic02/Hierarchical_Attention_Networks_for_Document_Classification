#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
from torch import nn


# In[9]:


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.context_vector = nn.Parameter(torch.randn(input_size))
        self.one_layer_MLP = nn.Linear(input_size, input_size)

    def forward(self, x):
        u = torch.tanh(self.one_layer_MLP(x))
        attention_weights = torch.softmax(torch.matmul(u, self.context_vector), dim=1)
        output = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)
        return output

