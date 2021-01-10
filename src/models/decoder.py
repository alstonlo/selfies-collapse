import torch
import torch.nn as nn


class DecoderRNN(nn.Module):

    def __init__(self, embedding, latent_dim,
                 hidden_dim, num_layers, n_vocab,
                 bos_idx, eos_idx, pad_idx):
        super(DecoderRNN, self).__init__()
        self.n_vocab = n_vocab
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.embedding = embedding
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(input_size=embedding.embedding_dim + latent_dim,
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

        # z -> (B, Z)
        x = nn.utils.rnn.pad_sequence(x, False, self.pad_idx)  # -> (L, B)
        seq_len, batch_size = x.size()

        x_i = x[0, :].unsqueeze(0)  # x_0 -> (1, B)
        h_i = torch.stack([self.fc_hidden(z)] * self.gru.num_layers, dim=0)
        # h_i -> (num layers, B, H)

        x_hat = torch.zeros(seq_len, batch_size, self.n_vocab,
                            device=x.device)  # -> (L, B, V)

        for i in range(1, seq_len):
            x_i = self.embedding(x_i)  # -> (1, B, E)
            z_i = z.unsqueeze(0)  # -> (1, B, Z)
            x_i = torch.cat([x_i, z_i], dim=-1)  # -> (1, B, E + Z)

            x_i, h_i = self.gru(x_i, h_i)  # -> (1, B, H), (num layers, B, H)
            o_i = self.fc_out(x_i)  # -> (1, B, O)

            x_hat[i] = o_i
            x_i = torch.argmax(o_i, dim=2)  # -> (1, B)

        return x_hat
