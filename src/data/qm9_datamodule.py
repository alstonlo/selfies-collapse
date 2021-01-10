import pathlib

import pytorch_lightning as pl
import selfies as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

from src.data.dataset import SELFIESDataset, SMILESDataset

QM9_PATH = pathlib.Path(__file__).parents[2] / 'datasets' / 'qm9.txt'


class QM9DataModule(pl.LightningDataModule):

    def __init__(self,
                 encoding, batch_size,
                 split_ratio=(0.8, 0.1, 0.1), split_seed=None):
        super().__init__()
        self.encoding = encoding
        assert encoding in ('smiles', 'selfies')
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.split_seed = split_seed

        self.dataset = None
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        with open(QM9_PATH, 'r') as f:
            lines = list(map(lambda s: s.rstrip('\n'),
                             f.readlines()))

        if self.encoding == 'smiles':
            self.dataset = SMILESDataset(lines)

        else:
            lines = list(map(sf.encoder, lines))
            self.dataset = SELFIESDataset(lines)

    def setup(self, stage=None):
        train_len = int(len(self.dataset) * self.split_ratio[0])
        val_len = int(len(self.dataset) * self.split_ratio[1])
        test_len = len(self.dataset) - train_len - val_len
        lengths = [train_len, val_len, test_len]

        generator = None
        if self.split_seed is not None:
            generator = torch.Generator().manual_seed(self.split_seed)

        self.train, self.val, self.test = \
            data.random_split(self.dataset, lengths, generator)

    def train_dataloader(self):
        return data.DataLoader(self.train,
                               batch_size=self.batch_size,
                               shuffle=True,
                               collate_fn=self._collate_fn)

    def val_dataloader(self):
        return data.DataLoader(self.val,
                               batch_size=self.batch_size,
                               collate_fn=self._collate_fn)

    def _collate_fn(self, sequences):
        return pad_sequence(sequences, batch_first=False,
                            padding_value=self.dataset.get_pad_idx())
