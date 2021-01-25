import torch
import torch.nn as nn


class EncoderRNN(nn.Module):

    def __init__(self, embedding, hidden_dim, num_layers, latent_dim):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = embedding
        self.gru = nn.GRU(input_size=embedding.embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=False)

        self.fc_mu = nn.Linear(num_layers * hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(num_layers * hidden_dim, latent_dim)

    def forward(self, x):
        # B = batch size
        # L = sequence length
        # E = embed dim
        # H = hidden dim
        # Z = latent dim
        # N = num layers

        x = [self.embedding(x_i) for x_i in x]
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        _, h_f = self.gru(x)  # -> (N, B, H)
        h = torch.cat(h_f.split(1), dim=-1).squeeze(0)  # -> (B, H * N)

        mu = self.fc_mu(h)  # -> (B, Z)
        log_var = self.fc_log_var(h)  # -> (B, Z)
        return mu, log_var
