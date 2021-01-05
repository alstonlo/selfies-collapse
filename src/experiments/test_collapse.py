import itertools
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

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
    betas = [1, 0.1, 0.01, 0.001, 0.0001]

    for encoding, beta in itertools.product(encodings, betas):

        model_name = f"{encoding}_beta={beta}"
        qm9 = qm9_dict[encoding]

        model = VAE(**vars(args),
                    beta=beta,
                    vocab_size=len(qm9.dataset.stoi),
                    sos_idx=qm9.dataset.get_sos_idx(),
                    pad_idx=qm9.dataset.get_pad_idx())

        logger = pl_loggers.TensorBoardLogger(
            log_dir, name=model_name, default_hp_metric=False
        )

        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            max_epochs=1,
            gradient_clip_val=args.grad_clip
        )
        trainer.fit(model, datamodule=qm9)
