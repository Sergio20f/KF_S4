
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch.distributions.normal import Normal
import copy
from torch.nn.parameter import Parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first):
        super().__init__()
        # Single-layer LSTM for the block
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=batch_first)
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, hidden_size)
        residual = self.residual_proj(x)        # (batch, seq_len, hidden_size)
        out = self.layer_norm(lstm_out + residual)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, num_classes):
        super().__init__()
    
        self.blocks = nn.ModuleList()
        self.blocks.append(LSTM_Block(input_size, hidden_size, batch_first))
        for _ in range(num_layers - 1):
            self.blocks.append(LSTM_Block(hidden_size, hidden_size, batch_first))
        
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        for block in self.blocks:
            x = block(x)  # (batch, seq_len, hidden_size)
        last_output = x[:, -1, :]  # (batch, hidden_size)
        logits = self.linear(last_output)
        return logits


class LinearRNN_Block(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.linear_x = nn.Linear(input_size, hidden_size) # B
        self.linear_h = nn.Linear(hidden_size, hidden_size, bias=False) # A
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Assume x has shape (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        h = None
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]  # (batch, input_size)
            if h is None:
                h = self.linear_x(xt)
            else:
                h = self.linear_x(xt) + self.linear_h(h)
            outputs.append(h.unsqueeze(1))
        # (batch, seq_len, hidden_size)
        out_seq = torch.cat(outputs, dim=1)
        residual = self.residual_proj(x)
        out = self.layer_norm(out_seq + residual)

        return out

class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(LinearRNN_Block(input_size, hidden_size, batch_first=batch_first))
        for _ in range(num_layers - 1):
            self.blocks.append(LinearRNN_Block(hidden_size, hidden_size, batch_first=batch_first))
        
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        last_output = x[:, -1, :]
        logits = self.linear(last_output)
        return logits


def init_reservoir_matrix(hidden_size):
    W = torch.randn(hidden_size, hidden_size)
    Q, R = torch.linalg.qr(W)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

class ReservoirLinearRNN_Block(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.linear_x = nn.Linear(input_size, hidden_size)
        # Create a fixed reservoir matrix and register it as a buffer so it is not updated during training.
        reservoir_matrix = init_reservoir_matrix(hidden_size)
        self.register_buffer('reservoir', reservoir_matrix) # A (untrained)
    
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        h = None
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]  # (batch, input_size)
            if h is None:
                h = self.linear_x(xt)
            else:
                h = self.linear_x(xt) + torch.matmul(h, self.reservoir)
            outputs.append(h.unsqueeze(1))
        # (batch, seq_len, hidden_size)
        out_seq = torch.cat(outputs, dim=1)
        residual = self.residual_proj(x)
        out = self.layer_norm(out_seq + residual)
        return out

class ReservoirLinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ReservoirLinearRNN_Block(input_size, hidden_size, batch_first=batch_first))
        for _ in range(num_layers - 1):
            self.blocks.append(ReservoirLinearRNN_Block(hidden_size, hidden_size, batch_first=batch_first))
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        last_output = x[:, -1, :]
        logits = self.linear(last_output)
        return logits