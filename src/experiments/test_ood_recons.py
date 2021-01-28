import pathlib
import random
from argparse import ArgumentParser

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from src.data.qm9_datamodule import QM9DataModule
from src.models.vae import VAE

# ------------
# file paths
# ------------
LOG_DIR = pathlib.Path(__file__).parents[2] / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_trial(trial: optuna.trial.Trial):
    # =================
    # model params
    # =================
    embed_dim = trial.suggest_int("embed_dim", low=25, high=200)
    enc_num_layers = trial.suggest_int("enc_num_layers", low=1, high=3)
    enc_hidden_dim = trial.suggest_int("enc_hidden_dim", low=25, high=200)
    latent_dim = trial.suggest_int("latent_dim", low=25, high=150)
    dec_num_layers = trial.suggest_int("dec_num_layers", low=1, high=3)
    dec_hidden_dim = trial.suggest_int("dec_hidden_dim", low=25, high=200)

    # ==================
    # training params
    # ==================
    split_seed = random.randint(100, 200)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    patience = trial.suggest_int("patience", low=1, high=10)
    lr = trial.suggest_loguniform("lr", low=1e-5, high=1e-3)
    beta = trial.suggest_loguniform("beta", low=1e-3, high=10)

    # ---------------------
    # main experiment
    # ---------------------
    for language in ['smiles', 'deep_smiles', 'selfies']:

        # ------------
        # data
        # ------------
        qm9 = QM9DataModule(
            language,
            batch_size=batch_size,
            split_seed=split_seed,
        )
        qm9.prepare_data()

        model = VAE(
            vocab=qm9.vocab,
            embed_dim=embed_dim,
            enc_hidden_dim=enc_hidden_dim,
            enc_num_layers=enc_num_layers,
            latent_dim=latent_dim,
            dec_hidden_dim=dec_hidden_dim,
            dec_num_layers=dec_num_layers,
            lr=lr,
            beta=beta,
        )

        logger = TensorBoardLogger(
            LOG_DIR, name=f"vae_{language}",
            default_hp_metric=False
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=patience,
            verbose=True,
            mode='min'
        )

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping],
            deterministic=True,
            progress_bar_refresh_rate=0,
            weights_summary=None,
            gpus=int(torch.cuda.is_available())
        )

        trainer.fit(model, datamodule=qm9)
        trainer.test(model, datamodule=qm9)

    return 1


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=299)
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()

    pl.seed_everything(seed=args.seed)

    study = optuna.create_study(
        study_name="ood_recons",
        storage="sqlite:///ood_recons.db",
        sampler=optuna.samplers.RandomSampler(seed=args.seed),
        load_if_exists=True
    )

    study.optimize(
        func=run_trial,
        n_trials=args.n_trials,
        timeout=14400,
        show_progress_bar=True
    )
