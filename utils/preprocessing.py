#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import pandas as pd
from collections import Counter
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt_tab')


# In[2]:


def tokenize_docs(df):
    tokenized_docs = []
    for doc in df.iloc[:,1]:
        sentences = sent_tokenize(doc)
        tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
        tokenized_docs.append(tokenized_sentences)
    return tokenized_docs


# In[16]:


def build_vocabulary(tokenized_docs):
    
    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    
    words = [word for doc in tokenized_docs for sentence in doc for word in sentence]
    word_count = Counter(words)
    
    for word, count in word_count.items():
        if count > 5:
            vocabulary[word] = len(vocabulary)
                
    return vocabulary, word_count


# In[24]:


def make_embedding_matrix(tokenized_docs, vocabulary, embedding_size=200):
    sentences = [sentence for doc in tokenized_docs for sentence in doc]
    
    model = Word2Vec(sentences=sentences, vector_size=embedding_size, min_count=5 ,workers=5, epochs=15)
    
    embedding_matrix = torch.zeros((len(vocabulary), embedding_size))
    embedding_matrix[vocabulary['<UNK>']] = torch.randn(embedding_size)
    for word, idx in vocabulary.items():
        if word in model.wv:
            embedding_matrix[idx] = torch.tensor(model.wv[word])
            
    return embedding_matrix


# In[5]:


def max_sentence_length(tokenized_docs):
    return max([len(sentence) for doc in tokenized_docs for sentence in doc])


# In[6]:


def max_document_length(tokenized_docs):
    return max([len(doc) for doc in tokenized_docs])


# In[17]:


def insert_padding(tokenized_docs, max_sentence_len, max_document_len):
    
    padded_docs = []
    for doc in tokenized_docs:
        padded_sentences = []
        for sentence in doc[:max_document_len]:
            padded_sentence = sentence[:max_sentence_len] + ['<PAD>']*(max_sentence_len - len(sentence))
            padded_sentences.append(padded_sentence)
        for _ in range(max_document_len - len(padded_sentences)):
            padded_sentences.append(['<PAD>']*max_sentence_len)
        padded_docs.append(padded_sentences)
        
    return padded_docs


def truncate(tokenized_docs, max_sentence_len, max_document_len):

    truncated_docs = []
    for doc in tokenized_docs:
        truncated_doc = []
        for sentence in doc:
            truncated_sentence = sentence[:max_sentence_len]
            truncated_doc.append(truncated_sentence)
        truncated_docs.append(truncated_doc[:max_document_len])

    return truncated_docs


# In[8]:


def replace_unk(tokenized_docs, word_count):
    new_docs = []
    for doc in tokenized_docs:
        new_doc = []
        for sentence in doc:
            new_sentence = [word if word_count[word] > 5 else '<UNK>' for word in sentence]
            new_doc.append(new_sentence)
        new_docs.append(new_doc)
        
    return new_docs


# In[23]:


def word_to_indices(tokenized_docs, vocabulary):
    ind_docs = []

    for doc in tokenized_docs:
        ind_doc = []
        for sentence in doc:
            ind_sentence = [vocabulary.get(word, 1) for word in sentence]
            ind_doc.append(ind_sentence)
        ind_docs.append(ind_doc)
    ind_docs = torch.tensor(ind_docs, dtype=torch.long)

    return ind_docs

