import pathlib
from typing import Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils import data

from src.data.dataset import (
    DeepSMILESVocab,
    LineByLineDataset,
    SELFIESVocab,
    SMILESVocab,
    Vocab
)

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
        assert language in ('smiles', 'deep_smiles', 'selfies')
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
            self.vocab = SMILESVocab(lines)

        elif self.language == 'deep_smiles':
            self.vocab = DeepSMILESVocab(lines)

        elif self.language == 'selfies':
            self.vocab = SELFIESVocab(lines)

        else:
            raise ValueError()

    def setup(self, stage=None):

        qm9 = pd.read_csv(QM9_PATH, header=0)

        if (stage == 'fit') or (stage is None):
            train_val_df = qm9[qm9['num_heavy_atoms'] <= 7]
            dataset = LineByLineDataset(
                train_val_df[self.language].tolist(),
                self.vocab
            )

            train_len = int(len(dataset) * self.split_ratio[0])
            val_len = len(dataset) - train_len

            generator = self._split_generator()
            self.train, self.val = data.random_split(
                dataset=dataset,
                lengths=[train_len, val_len],
                generator=generator
            )

        if (stage == 'test') or (stage is None):
            test_df = qm9[qm9['num_heavy_atoms'] == 9]
            dataset = LineByLineDataset(
                test_df[self.language].tolist(),
                self.vocab
            )

            generator = self._split_generator()
            self.test, _ = data.random_split(
                dataset=dataset,
                lengths=[10000, len(dataset) - 10000],
                generator=generator
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

    def _split_generator(self):
        if self.split_seed is None:
            return None
        else:
            return torch.Generator().manual_seed(self.split_seed)

    @staticmethod
    def _collate_fn(batch):
        return list(batch)
