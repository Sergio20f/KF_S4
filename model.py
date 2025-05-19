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
    
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Final MLP
        self.mlp = MLP(hidden_size, mlp_hidden_dim, hidden_size, mlp_num_layers)
        
    def forward(self, x, y_KF=None, R=None, Sigma_pred=None, c=5.0):
        # (batch, seq_len, input_size)
        batch, seq_len, nfeat = x.shape
        wt_list = []
        wt_cont_list = []
        wt_update_list = []
        err_list = []

        if y_KF is None:
            h = None
            outputs = []

            for t in range(seq_len):
                xt = x[:, t, :]  # (batch, input_size)
                u_t = self.encoder(xt) # (batch, M)
                Bu_t = self.B(u_t) # (batch, H)

                if h is None:
                    h = Bu_t
                else:
                    # err = h - torch.matmul(h, self.A) - self.B(self.encoder(x[:, t-1, :])) # h_t-1 - Ah_t-1 - Bu_{t-1}
                    h = torch.matmul(h, self.A) + Bu_t

                h_prev = h
                outputs.append(h.unsqueeze(1))

        else:
            P_post = Sigma_pred.expand(1, -1, -1)
            qt = 1.0
            Q = torch.eye(self.hidden_size, device=device) * qt
            BQB = self.B.weight @ self.B.weight.transpose(-1, -2) * qt # is hparam? # Q.expand(1, -1, -1)
            
            w_last = 1.0
            theta_pred = None
            h_pred = None

            outputs = []

            for t in range(seq_len):
                if t == 0: # TEMP
                    x_prev = torch.zeros_like(x[:, 0, :])
                    u_prev = self.encoder(x_prev) # (batch, M)
                    Bu_tilde = self.B(u_prev)
                else:
                    x_prev = x[:, t-1, :]  # (batch, input_size)

                Bu_prev = Bu_tilde

                # u_t
                u_t = self.encoder(x[:, t, :])
                Bu_t = self.B(u_t)

                # Predict Step for θ
                if theta_pred is None:
                    theta_pred = Bu_prev # We want -> (hidden_size)
                    h_t_noB = Bu_t
                else:
                    theta_pred = torch.matmul(theta_pred, self.A) + Bu_prev # We want -> (hidden_size)
                    h_t_noB = torch.matmul(h_pred, self.A)

                # Predict covariance
                P_pred = self.A @ P_post @ self.A.transpose(-1, -2) + Q # TODO: + torch.eye(self.hidden_size, device=x.device) * qt -- wrong?

                # TODO: try Mahalanobis distance
                # Innovation -- Measurement: r_t = h_t - Bu_t = y[:, t, :] - B(self.encoder(x[:, t, :])) = A theta_t - A theta_pred
                if t == 0:
                    err = h_t_noB - torch.matmul(theta_pred, self.A) # Ah_{t-1} - A\theta_t
                else:
                    err = torch.matmul(h_pred, self.A) - torch.matmul(h_pred, torch.linalg.matrix_power(self.A, 2)) - Bu_prev
                err_downdate = Bu_t - Bu_prev

                wt_cont = 1 / torch.sqrt(1 + torch.linalg.norm(err_downdate) ** 2 / c**2) # WoLF
                wt = torch.tensor([1.0], device=device) if wt_cont >= 0.1 else torch.tensor([0.], device=device)
                
                weighted_BQB = BQB if wt == 1.0 else torch.eye(len(BQB)).to(device) * 1e3

                S = (
                    self.A @ P_pred @ self.A.transpose(-1, -2)
                    + weighted_BQB
                    # + BQB / wt
                )

                # Kalman Gain
                gain_rhs = P_pred @ self.A.transpose(-1, -2)
                X_T = torch.linalg.lstsq(S, gain_rhs.transpose(-1, -2))[0]
                K = X_T.transpose(-1, -2).expand(1, -1, -1)

                # Update Step
                theta_pred = theta_pred + torch.matmul(err, K)
                theta_pred = theta_pred.squeeze(0)

                P_pred = P_pred - K @ S @ K.transpose(-1, -2)

                # Mapping to h space
                Bu_tilde = wt*Bu_t + (1 - wt)*Bu_prev
                h_pred = torch.matmul(theta_pred, self.A) + Bu_tilde
                outputs.append(h_pred.unsqueeze(1))
                
                err_update = h_t_noB - torch.matmul(theta_pred, self.A)
                wt_update = 1 / torch.sqrt(1 + torch.linalg.norm(err_update) ** 2 / c**2) # WoLF

                # For plotting purposes
                wt_list.append(wt.item())
                wt_cont_list.append(wt_cont.item())
                wt_update_list.append(wt_update.item())
                # Little cheatcode for the plot to be clearer
                if t == 0:
                    err_list.append(torch.tensor([0.0], device=device).item())
                else:
                    err_list.append((torch.linalg.norm(err)**2).item())
                    print(f"Error norm: {torch.linalg.norm(err).item()}")
            num_zeros = sum(1 for wt in wt_list if wt == 0)
            print(f"Number of zeros in wt list: {num_zeros}")
        
        # (batch, seq_len, hidden_size)
        H_seq_pre = torch.cat(outputs, dim=1)
        residual = self.residual_proj(x) # (batch, seq_length, hidden_size)
        H_seq = self.layer_norm(H_seq_pre + residual) # (1, 50, 128) + (128) i.e. C = I

        h_last = H_seq[:, -1, :] # (batch, hidden_size)
        out = self.mlp(h_last)
        return out, H_seq_pre, (wt_list, wt_update_list, wt_cont_list, err_list)

class ReservoirLinearRNN(nn.Module):
    def __init__(self, input_size, encode_dim, hidden_size, mlp_hidden_dim, mlp_num_layers, num_layers, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList()

        for _ in range(num_layers):
            self.blocks.append(
                ReservoirLinearRNN_Block(
                    input_size=input_size,
                    encode_dim=encode_dim,
                    hidden_size=hidden_size,
                    mlp_hidden_dim=mlp_hidden_dim,
                    mlp_num_layers=mlp_num_layers,
                )
            )
            input_size = hidden_size  # so next block sees dimension H

        self.final_linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, y_KF=None, R=None, Sigma_pred=None):
        y_KF_list = [] # List of y's for each block (each element of the list is a tensor of shape (batch, seq_len, hidden_size))
        w_lists = []
        for block in self.blocks:
            x, y_KF_val, w_sublist = block(x, y_KF=y_KF, R=R, Sigma_pred=Sigma_pred)
            y_KF_list.append(y_KF_val)
            w_lists.append(w_sublist)
        last_output = x#[:, -1, :]
        logits = self.final_linear(last_output)

        return logits, y_KF_list, w_lists

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
