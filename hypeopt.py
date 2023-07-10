import optuna
from datasetload import train_loader, valid_loader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from hpolitmodel import LitModel
import torch.nn as nn


# TODO: hyperparameter alert depth of Transformer Blocks
# TODO: hyperparameter alert embedding dimensions
# TODO: hyperparameter alert number of heads
# todo: hyperparameter alert decide on number of hidden layers in feed forward
# todo: hyperparameter alert learning rate
# todo: hyperparamater alert number of epochs


def define_model_config(trial):
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    depth = trial.suggest_int("depth", 4, 36, step=2)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    num_heads = trial.suggest_int('headbase', 4, 16, step=4)  # Define a common base value
    factor_emb = trial.suggest_int('factor_emb', 32, 64, step=2)  # Multiplier for the base to define the embedding size
    embedding_dim = num_heads * factor_emb


    model_config = {"embedding_dim": embedding_dim,
                    "dropout": dropout,
                    "depth": depth,
                    "head": num_heads,
                    "pi": "base",
                    "weight_decay": weight_decay,
                    "lr": lr
                    }

    return model_config


def objective(trial: optuna.trial.Trial):
    logger = CSVLogger(version="trial"+str(trial.number), save_dir="/home/igardner/hpologsnew", name="hpotrials")
    model_config = define_model_config(trial)
    model = LitModel(model_config)
    trainer = pl.Trainer(max_epochs=10, logger=logger,  enable_checkpointing=False, enable_progress_bar=False,
                         check_val_every_n_epoch=10, limit_val_batches=10, limit_train_batches=50, strategy='ddp_find_unused_parameters_true')
    trainer.logger.log_hyperparams(model_config)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="SiDBTransformer_Hyperparameters")
    study.optimize(objective, n_trials=200, timeout=14400, gc_after_trial=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv("/home/igardner/hpologsnew/hpotrials.csv")
    print(study.best_trial)
    best_params = study.best_params
    # Assuming study is your Optuna study object
    fig = optuna.visualization.plot_parallel_coordinate(study, target_name = "val_acc")
    # Save the figure
    # pip install -U kaleido
    fig.write_image("/home/igardner/hpologsnew/optimization_history.png")
