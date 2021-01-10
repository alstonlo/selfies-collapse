import pathlib
from typing import Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils import data

from src.data.dataset import LineByLineDataset, Vocab

QM9_PATH = pathlib.Path(__file__).parents[2] / 'datasets' / 'qm9' / 'qm9.csv'


class QM9DataModule(pl.LightningDataModule):
    language: str
    batch_size: int
    split_ratio: Tuple[int, int]
    split_seed: Optional[int]

    vocab: Optional[Vocab]
    train: Optional[LineByLineDataset]
    val: Optional[LineByLineDataset]
    test: Optional[LineByLineDataset]

    def __init__(self,
                 language: str,
                 batch_size: int,
                 split_ratio: Tuple[int, int] = (0.9, 0.1),
                 split_seed: Optional[int] = None):
        super().__init__()
        self.language = language
        assert language in ('smiles', 'selfies')
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.split_seed = split_seed

        self.vocab = None
        self.train = None
        self.val = None
        self.test = None

    def prepare_data(self):
        qm9_df = pd.read_csv(QM9_PATH, header=0)

        lines = qm9_df[self.language].tolist()
        if self.language == 'smiles':
            self.vocab = Vocab.build_from_smiles(lines)
        else:
            self.vocab = Vocab.build_from_selfies(lines)

    def setup(self, stage=None):

        qm9 = pd.read_csv(QM9_PATH, header=0)

        if (stage == 'fit') or (stage is None):
            qmonly9 = qm9[qm9['num_heavy_atoms'] == 9]
            dataset = LineByLineDataset(
                qmonly9[self.language].tolist(),
                self.vocab
            )

            train_len = int(len(dataset) * self.split_ratio[0])
            val_len = len(dataset) - train_len

            generator = None if self.split_seed is None \
                else torch.Generator().manual_seed(self.split_seed)
            self.train, self.val = data.random_split(
                dataset=dataset,
                lengths=[train_len, val_len],
                generator=generator
            )

        if (stage == 'test') or (stage is None):
            qm7 = qm9[qm9['num_heavy_atoms'] <= 7]
            self.test = LineByLineDataset(
                qm7[self.language].tolist(),
                self.vocab
            )

    def train_dataloader(self):
        return data.DataLoader(self.train,
                               batch_size=self.batch_size,
                               shuffle=True,
                               collate_fn=self._collate_fn)

    def val_dataloader(self):
        return data.DataLoader(self.val,
                               batch_size=self.batch_size,
                               collate_fn=self._collate_fn)

    def test_dataloader(self):
        return data.DataLoader(self.test,
                               batch_size=self.batch_size,
                               collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(batch):
        return list(batch)
