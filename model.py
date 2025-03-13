import torch
import torch.nn as nn
from helpers import init_reservoir_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        
        elif num_layers == 0:
            layers.append(nn.Identity())

        else:
            # First layer: from input_dim to hidden_dim.
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Middle layers.
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                
            # Final layer: from hidden_dim to output_dim.
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class ReservoirLinearRNN_Block(nn.Module):
    def __init__(self, input_size, encode_dim, hidden_size, mlp_hidden_dim, mlp_num_layers):
        """
        A single block that does:
        1) Linear encoder from input dim L to some intermediate dim M
        2) Multiply by B to map to hidden dim H
        3) Reservoir update with a fixed or partially fixed matrix A (diag(lambda) or random)
        4) Residual + layer norm
        5) MLP
        """
        super().__init__()

        # Params
        self.hidden_size = hidden_size

        self.encoder = nn.Linear(input_size, encode_dim) if input_size != encode_dim else nn.Identity() # First linear layer
        self.B = nn.Linear(encode_dim, hidden_size, bias=False)

        # Create a fixed reservoir matrix and register it as a buffer so it is not updated during training.
        A = init_reservoir_matrix(hidden_size)
        self.register_buffer('A', A)

        self.C = nn.Identity() # C = I; for now
    
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Final MLP
        self.mlp = MLP(hidden_size, mlp_hidden_dim, hidden_size, mlp_num_layers)
        
    def forward(self, x, y_KF=None, R=None, Sigma_pred=None):
        if not y_KF:
            # (batch, seq_len, input_size)
            batch, seq_len, _ = x.shape
            h = None
            outputs = []

            for t in range(seq_len):
                xt = x[:, t, :]  # (batch, input_size)
                u_t = self.encoder(xt) # (batch, M)
                Bu_t = self.B(u_t) # (batch, H)

                if h is None:
                    h = Bu_t
                else:
                    h = torch.matmul(h, self.A) + Bu_t

                outputs.append(h.unsqueeze(1))

        else:
            y = y_KF
            h_pred = None
            outputs = []

            for t in range(seq_len):
                xt = x[:, t, :]  # (batch, input_size)
                u_t = self.encoder(xt)
                Bu_t = self.B(u_t)

                # Predict Step
                if h_pred is None:
                    h_pred = Bu_t
                else:
                    h_pred = torch.matmul(h, self.A) + Bu_t

                Sigma_pred = self.A @ Sigma_pred @ self.A.transpose(-1, -2)

                # Innovation
                err = y - self.C @ h_pred
                S = self.C @ Sigma_pred @ self.C.transpose(-1, -2) + self.R

                # Kalman Gain
                K = torch.linalg.solve(S, self.C @ Sigma_pred).transpose(-1, -2) # TODO: Is Sigma_pred.T or no?

                # Update Step
                h = h_pred + K @ err
                Sigma_pred = Sigma_pred - K @ S @ K.transpose(-1, -2)
                outputs.append(h.unsqueeze(1))
            
        # (batch, seq_len, hidden_size)
        H_seq = torch.cat(outputs, dim=1)
        residual = self.residual_proj(x)
        H_seq = self.layer_norm(H_seq + residual) # i.e. C = I

        h_last = H_seq[:, -1, :] # (batch, hidden_size)
        out = self.mlp(h_last)
        return out

class ReservoirLinearRNN(nn.Module):
    def __init__(self, input_size, encode_dim, hidden_size, mlp_hidden_dim, mlp_num_layers, num_layers, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList()

        for _ in range(num_layers):
            self.blocks.append(
                ReservoirLinearRNN_Block(
                    input_dim=input_size,
                    encode_dim=encode_dim,
                    hidden_dim=hidden_size,
                    mlp_hidden_dim=mlp_hidden_dim,
                    mlp_num_layers=mlp_num_layers,
                )
            )
            input_size = hidden_size  # so next block sees dimension H

        self.final_linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        y_KF = [] # List of y's for each block (each element of the list is a tensor of shape (batch, seq_len, hidden_size))
        for block in self.blocks:
            x = block(x)
            y_KF.append(x)
        last_output = x[:, -1, :]
        logits = self.final_linear(last_output)
        return logits, y_KF

###################################################### BASELINES #################################################################

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
        self.B = nn.Linear(input_size, hidden_size) # B
        self.A = nn.Linear(hidden_size, hidden_size, bias=False) # A
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
                h = self.B(xt)
            else:
                h = self.B(xt) + self.A(h)
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
