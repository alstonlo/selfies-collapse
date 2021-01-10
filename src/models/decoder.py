import torch
import torch.nn as nn


class DecoderRNN(nn.Module):

    def __init__(self, embedding, latent_dim,
                 hidden_dim, num_layers, n_vocab,
                 eos_idx):
        super(DecoderRNN, self).__init__()
        self.eos_idx = eos_idx

        self.embedding = embedding
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(input_size=embedding.embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=False)
        self.fc_out = nn.Linear(hidden_dim, n_vocab)

    def forward(self, z, x):
        # B = batch size
        # L = sequence length
        # E = embed dim
        # H = hidden dim
        # Z = latent dim
        # V = vocab size

        # z -> (B, Z),  x -> (L, B)
        seq_len, batch_size = x.size()

        x_hat = []
        x_i = self.eos_idx \
              * torch.ones((1, batch_size), dtype=torch.long, device=x.device)
        h_i = torch.stack([self.fc_hidden(z)] * self.gru.num_layers, dim=0)
        # x_i -> (1, B);  h_i -> (num layers, B, H)

        for i in range(seq_len):
            x_i = self.embedding(x_i)  # -> (1, B, E)
            x_i, h_i = self.gru(x_i, h_i)  # -> (1, B, H), (num layers, B, H)
            o_i = self.fc_out(x_i)  # -> (1, B, O)

            x_hat.append(o_i)
            x_i = torch.argmax(o_i, dim=2)  # -> (1, B)

        return torch.cat(x_hat, dim=0)  # -> (L, B, V)
