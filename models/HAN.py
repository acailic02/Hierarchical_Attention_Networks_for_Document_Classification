#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
from torch import nn
from torch.nn import functional as F 
import sys
sys.path.append(r'C:\Users\geass\Hierarchical_Attention_Networks_for_Document_Classification\models')
from Attention import Attention
from BiGRUEncoder import BiGRUEncoder


# In[19]:


class HAN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size):
        super(HAN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=0)
        embedding_dim = embedding_matrix.size(1)
        self.word_encoder = BiGRUEncoder(embedding_dim, hidden_size)
        self.word_attention = Attention(2*hidden_size)
        self.sentence_encoder = BiGRUEncoder(2*hidden_size, hidden_size)
        self.sentence_attention = Attention(2*hidden_size)
        self.linear = nn.Linear(2*hidden_size, output_size)
    

    def forward(self, x):
        batch_size, num_sentences, num_words = x.size()
        x = x.view(batch_size * num_sentences, num_words)
        x = self.embedding(x)

        x = self.word_encoder(x)
        x = self.word_attention(x)

        x = x.view(batch_size, num_sentences, -1)
        x = self.sentence_encoder(x)
        x = self.sentence_attention(x)

        logits = self.linear(x)
        probabilities = F.log_softmax(logits, dim=1)
        
        return probabilities

