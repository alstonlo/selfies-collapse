from argparse import ArgumentParser

import pytorch_lightning as pl

from src.data.qm9_datamodule import QM9DataModule
from src.models.vae import VAE

import pathlib

def experiment_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument('--encoding', type=str, default='selfies')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.)

    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    qm9 = QM9DataModule(encoding=args.encoding,
                        batch_size=args.batch_size)
    qm9.prepare_data()

    # ------------
    # model
    # ------------
    model = VAE(**vars(args),
                vocab_size=len(qm9.dataset.stoi),
                sos_idx=qm9.dataset.get_sos_idx(),
                pad_idx=qm9.dataset.get_pad_idx())

    # ------------
    # training
    # ------------
    save_dir = pathlib.Path(__file__).parents[1] / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=args.min_delta,
                                                patience=args.patience,
                                                verbose=True,
                                                mode='min')

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[early_stopping],
        default_root_dir=save_dir,
        gradient_clip_val=args.grad_clip,
        progress_bar_refresh_rate=5,
        auto_lr_find=True
    )
    trainer.fit(model, datamodule=qm9)


if __name__ == '__main__':
    experiment_main()
