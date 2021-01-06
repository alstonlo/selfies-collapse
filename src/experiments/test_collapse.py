import itertools
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data.qm9_datamodule import QM9DataModule
from src.models.vae import VAE


def test_collapse_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=1.)

    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    qm9_dict = {
        'smiles': QM9DataModule('smiles', args.batch_size, split_seed=12),
        'selfies': QM9DataModule('smiles', args.batch_size, split_seed=12)
    }
    qm9_dict['smiles'].prepare_data()
    qm9_dict['selfies'].prepare_data()

    # ------------
    # file paths
    # ------------
    log_dir = pathlib.Path(__file__).parents[2] / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------
    # main experiment
    # ---------------------
    encodings = ['smiles', 'selfies']
    betas = [1, 5, 10]

    for encoding, beta in itertools.product(encodings, betas):

        qm9 = qm9_dict[encoding]

        model = VAE(**vars(args),
                    beta=beta,
                    vocab_size=len(qm9.dataset.stoi),
                    sos_idx=qm9.dataset.get_sos_idx(),
                    pad_idx=qm9.dataset.get_pad_idx())

        version = f"enc={encoding}_beta={beta}"
        logger = TensorBoardLogger(
            log_dir, name='vae_dec_XL', version=version,
            default_hp_metric=False
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode='min'
        )

        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=[early_stopping],
            gradient_clip_val=args.grad_clip
        )
        trainer.fit(model, datamodule=qm9)
