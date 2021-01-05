from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data

from src.models.decoder import DecoderRNN
from src.models.encoder import EncoderRNN


class VAE(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model arguments
        parser.add_argument('--embed_dim', type=int, default=50)
        parser.add_argument('--enc_hidden_dim', type=int, default=100)
        parser.add_argument('--latent_dim', type=int, default=50)
        parser.add_argument('--dec_hidden_dim', type=int, default=50)
        parser.add_argument('--dec_num_layers', type=int, default=2)

        return parser

    def __init__(self, embed_dim,
                 enc_hidden_dim,
                 latent_dim,
                 dec_hidden_dim, dec_num_layers,
                 vocab_size, pad_idx, sos_idx,
                 lr, beta, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=pad_idx)

        self.encoder = EncoderRNN(embedding=self.embedding,
                                  hidden_dim=enc_hidden_dim,
                                  latent_dim=latent_dim)

        self.decoder = DecoderRNN(embedding=self.embedding,
                                  latent_dim=latent_dim,
                                  hidden_dim=dec_hidden_dim,
                                  num_layers=dec_num_layers,
                                  out_dim=vocab_size,
                                  sos_idx=sos_idx)

        self.loss_f = nn.CrossEntropyLoss(ignore_index=pad_idx)

    @auto_move_data
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, x)

        return x_hat, z, mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.rand_like(std)
        return mu + epsilon * std

    def training_step(self, batch, batch_idx):
        x_hat, z, mu, log_var = self(batch)

        loss, (recons, kld) = self.compute_loss(x=batch, x_hat=x_hat,
                                                mu=mu, log_var=log_var)

        self.log('train_loss', loss)
        self.log('train_recons', recons)
        self.log('train_kld', kld)

        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, z, mu, log_var = self(batch)

        loss, (recons, kld) = self.compute_loss(x=batch, x_hat=x_hat,
                                                mu=mu, log_var=log_var)

        self.log('val_loss', loss)
        self.log('val_recons', recons)
        self.log('val_kld', kld)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def compute_loss(self, x, x_hat, mu, log_var):

        # Reconstruction Loss
        x_hat = x_hat.flatten(start_dim=0, end_dim=1)  # -> (B * L, E)
        x = x.flatten(start_dim=0, end_dim=1)  # -> (B * L,)
        recons = self.loss_f(x_hat, x)

        # KL[q(z|x)||p(z)]
        kld = torch.mean(torch.sum(
            0.5 * (log_var.exp() + mu.pow(2) - log_var - 1.0), dim=1
        ))

        loss = recons + self.hparams.beta * kld

        return loss, (recons, kld)
