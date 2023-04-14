from itertools import accumulate
import hydra
import wandb
import logging
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.utils import log_hyperparams

log = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def train(config: DictConfig):
    log.info(f"Instantiating logger <{config.logger._target_}>")
    
    # next two lines are a bit of a hacky way to register tags because the comet api is not great.
    logger: WandbLogger = hydra.utils.instantiate(config.logger)
    
    log.info(f"Instantiating trainer <{config.trainer._target_}>")


    lr_monitor = LearningRateMonitor(logging_interval=config.lr_log_interval)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[lr_monitor]
    )
    print(f'precision {trainer.precision}')
    

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info("Logging hyperparameters!")
    log_hyperparams(config=config, trainer=trainer)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)