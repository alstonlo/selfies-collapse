import pathlib
import random

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from src.data.qm9_datamodule import QM9DataModule
from src.models.vae import VAE

SEED = 299
N_TRIALS = 1
N_TEST_MOLS = 10000

# ------------
# file paths
# ------------
LOG_DIR = pathlib.Path(__file__).parents[2] / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_trial(trial: optuna.trial.Trial):
    # =================
    # model params
    # =================
    embed_dim = trial.suggest_int("embed_dim", low=50, high=100)
    enc_num_layers = trial.suggest_int("enc_num_layers", low=1, high=3)
    enc_hidden_dim = trial.suggest_int("enc_hidden_dim", low=25, high=100)
    latent_dim = trial.suggest_int("latent_dim", low=25, high=150)
    dec_num_layers = trial.suggest_int("dec_num_layers", low=1, high=3)
    dec_hidden_dim = trial.suggest_int("dec_hidden_dim", low=25, high=100)

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
    for lang in ['smiles', 'deep_smiles', 'selfies']:

        # ------------
        # data
        # ------------
        qm9 = QM9DataModule(
            lang,
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
            LOG_DIR, name=f"vae_{lang}",
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
            limit_test_batches=int(N_TEST_MOLS / batch_size),
            fast_dev_run=3,  # TODO: comment out
        )

        trainer.fit(model, datamodule=qm9)
        trainer.test(model, datamodule=qm9)

    return 1


def main():
    pl.seed_everything(SEED)

    study = optuna.create_study(
        study_name="ood_recons",
        sampler=optuna.samplers.RandomSampler(seed=SEED)
    )
    study.optimize(run_trial, n_trials=N_TRIALS)
