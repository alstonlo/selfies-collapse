import torch.nn as nn


class EncoderRNN(nn.Module):

    def __init__(self, embedding, hidden_dim, latent_dim):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = embedding
        self.gru = nn.GRU(input_size=embedding.embedding_dim,
                          hidden_size=hidden_dim,
                          batch_first=False)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # B = batch size
        # L = sequence length
        # E = embed dim
        # H = hidden dim
        # Z = latent dim

        # x -> (L, B)
        x = self.embedding(x)  # -> (L, B, E)
        _, h_final = self.gru(x)  # -> (1, B, H)
        h = h_final.squeeze()  # -> (B, H)

        mu = self.fc_mu(h)  # -> (B, Z)
        log_var = self.fc_log_var(h)  # -> (B, Z)
        return mu, log_var
