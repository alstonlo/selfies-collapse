import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from src.data.qm9_datamodule import QM9DataModule
from src.models.vae import VAE


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--n_test_mols', type=int, default=10000)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--beta', type=float, default=0.05)

    parser.add_argument('--seed', type=int, default=299)

    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # ------------
    # file paths
    # ------------
    log_dir = pathlib.Path(__file__).parents[2] / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------
    # main experiment
    # ---------------------
    for lang in ['smiles', 'selfies']:
        for i in range(args.n_trials):

            # ------------
            # data
            # ------------
            qm9 = QM9DataModule(lang, batch_size=args.batch_size)
            qm9.prepare_data()

            model = VAE(**vars(args), vocab=qm9.vocab)

            logger = TensorBoardLogger(
                log_dir, name=f"vae_{lang}", version=i,
                default_hp_metric=False
            )

            early_stopping = EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=5,
                verbose=True,
                mode='min'
            )

            trainer = pl.Trainer.from_argparse_args(
                args,
                logger=logger,
                callbacks=[early_stopping],
                deterministic=True,
                limit_test_batches=int(args.n_test_mols / args.batch_size)
            )

            trainer.fit(model, datamodule=qm9)
            trainer.test(model, datamodule=qm9)
