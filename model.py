import torch
import torch.nn as nn

class SRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.num_matrices = 3
        self.weight = nn.Parameter(torch.Tensor(input_size, self.hidden_size * self.num_matrices))
        self.bias = nn.Parameter(torch.Tensor(2 * self.hidden_size))
        self.v = nn.Parameter(torch.Tensor(2 * self.hidden_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        val_range = (3.0 / self.input_size) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()
        self.v.data.zero_()

    def forward(self, input):
        seq_len, batch_size, input_size = input.shape
        c0 = torch.zeros(batch_size, self.input_size, dtype=input.dtype,device=input.device)
        U = input @ self.weight
        # shape (seq_len, batch_size, 3 * hidden_size)
        U = U.view(seq_len, batch_size, self.hidden_size, self.num_matrices)
        U0 = U[..., 0]
        U1 = U[..., 1]
        U2 = U[..., 2]
        b0, b1 = self.bias.view(2, self.hidden_size)
        v0, v1 = self.bias.view(2, self.hidden_size)

        c_prev= c0
        H, C = [], []
        for t in range(seq_len):
            f_t = self.sigmoid(U0[t, :, :]+ v0 * c_prev + b0)
            r_t = self.sigmoid(U1[t, :, :]+ v1 * c_prev + b1)
            c_t = f_t * c_prev + (1 - f_t) * U2[t, :, :]
            h_t = r_t * c_t + (1 - r_t) * input[t, :, :]
            c_prev = c_t
            C.append(c_t)
            H.append(h_t)
        H = torch.stack(H)
        C = torch.stack(C)
        return H, C


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.num_layers = num_layers
        self.srus = nn.ModuleList([SRUCell(input_size=input_size, hidden_size=hidden_size) for _ in range(num_layers)])
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    

    def forward(self, input):
        h = input
        for _, sru in enumerate(self.srus):
            h, c = sru(h)
            h=self.linear(h)
        return h