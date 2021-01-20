from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data

from src.data.dataset import Vocab
from src.models.decoder import DecoderRNN
from src.models.encoder import EncoderRNN
from src.models.metrics import Accuracy, EditDistance


class VAE(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model arguments
        parser.add_argument('--embed_dim', type=int, default=100)
        parser.add_argument('--enc_hidden_dim', type=int, default=100)
        parser.add_argument('--latent_dim', type=int, default=50)
        parser.add_argument('--dec_hidden_dim', type=int, default=100)
        parser.add_argument('--dec_num_layers', type=int, default=1)

        return parser

    def __init__(self,
                 vocab: Vocab,
                 embed_dim: int,
                 enc_hidden_dim: int,
                 latent_dim: int,
                 dec_hidden_dim: int,
                 dec_num_layers: int,
                 lr: float,
                 beta: float,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.lr = lr
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=embed_dim,
                                      padding_idx=vocab.pad_idx)

        self.encoder = EncoderRNN(embedding=self.embedding,
                                  hidden_dim=enc_hidden_dim,
                                  latent_dim=latent_dim)

        self.decoder = DecoderRNN(embedding=self.embedding,
                                  latent_dim=latent_dim,
                                  hidden_dim=dec_hidden_dim,
                                  num_layers=dec_num_layers,
                                  n_vocab=len(vocab),
                                  bos_idx=vocab.bos_idx,
                                  eos_idx=vocab.eos_idx,
                                  pad_idx=vocab.pad_idx)

        self.loss_f = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

        self.test_acc = Accuracy(ignore_index=vocab.pad_idx)
        self.test_edit_dist = EditDistance(eos_idx=vocab.eos_idx)

    @auto_move_data
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, x)
        x = nn.utils.rnn.pad_sequence(x, False, self.vocab.pad_idx)

        return x, x_hat, z, mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.rand_like(std)
        return mu + epsilon * std

    def training_step(self, batch, batch_idx):
        x, x_hat, z, mu, log_var = self(batch)

        loss, (recons, kld) = \
            self.compute_loss(x=x, x_hat=x_hat, mu=mu, log_var=log_var)

        self.log('train_loss', loss)
        self.log('train_recons', recons)
        self.log('train_kld', kld)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, z, mu, log_var = self(batch)

        loss, (recons, kld) = \
            self.compute_loss(x=x, x_hat=x_hat, mu=mu, log_var=log_var)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recons', recons)
        self.log('val_kld', kld)

        return loss

    def test_step(self, batch, batch_idx):
        x, x_hat, z, mu, log_var = self(batch)

        loss, (recons, kld) = \
            self.compute_loss(x=x, x_hat=x_hat, mu=mu, log_var=log_var)

        x = x[1:].transpose(0, 1)  # remove sos token, and set batch_first=True
        x_hat = x_hat[1:].transpose(0, 1).argmax(dim=-1)

        self.test_acc(preds=x_hat, targets=x)
        self.test_edit_dist(preds=x_hat, targets=x)

        self.log('test_loss', loss)
        self.log('test_recons', recons)
        self.log('test_kld', kld)
        self.log('test_acc', self.test_acc)
        self.log('test_edit_dist', self.test_edit_dist)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def compute_loss(self, x, x_hat, mu, log_var):
        x = x[1:]  # remove sos token
        x_hat = x_hat[1:]

        # Reconstruction Loss
        x_hat = x_hat.view(-1, x_hat.size(2))  # -> (B * L, E)
        x = x.view(-1)  # -> (B * L,)
        recons = self.loss_f(x_hat, x)

        # KL[q(z|x)||p(z)]
        kld = torch.mean(torch.sum(
            0.5 * (log_var.exp() + mu.pow(2) - log_var - 1.0), dim=1
        ))

        loss = recons + self.hparams.beta * kld

        return loss, (recons, kld)
