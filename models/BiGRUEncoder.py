#!/usr/bin/env python
# coding: utf-8

# In[29]:


import torch
from torch import nn
import numpy as np


# In[34]:


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)

        self.Uz = nn.Linear(hidden_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.shape
        h_prev = torch.zeros(batch_size, self.hidden_size, device=input_seq.device)
            
        hs = []
        for t in range(seq_len):
            x_t = input_seq[:,t,:]     # taking t-th time step
            update_gate = torch.sigmoid(self.Wz(x_t) + self.Uz(h_prev))
            reset_gate = torch.sigmoid(self.Wr(x_t) + self.Ur(h_prev))
            candidate_state = torch.tanh(self.Wh(x_t) + reset_gate * self.Uh(h_prev))
            h_t = (1 - update_gate) * h_prev + update_gate * candidate_state
            hs.append(h_t)
            h_prev = h_t
            
        hs = torch.stack(hs, dim=1)
        
        return hs


# In[35]:


class BiGRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUEncoder, self).__init__()
        self.forward_gru = GRU(input_size, hidden_size)
        self.backward_gru = GRU(input_size, hidden_size)

    def forward(self, input_seq):
        # input_seq dim = [batch, seq_length, input_size]
        h_forward = self.forward_gru(input_seq)
        input_reversed = torch.flip(input_seq, dims=[1])
        h_backward = torch.flip(self.backward_gru(input_reversed), dims=[1])
        output = torch.cat([h_forward, h_backward], dim=2)
        return output

