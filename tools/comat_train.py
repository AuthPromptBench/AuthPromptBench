import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from models.comat import CoMat
from data.dataset import GanDataModule
from models.utils.logger import get_logger
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import wandb
wandb.login(key="YOUR_WANDB_API_KEY")

logger = get_logger(__name__)

def main(gpus: int, nodes: int, config_path: str = "configs/config.yaml"):
    # Load configuration
    config = OmegaConf.load(config_path)
    kwargs = OmegaConf.to_container(config, resolve=True)
    kwargs["model"]["do_classifier_free_guidance"] = kwargs["model"]["cfg_scale"] > 1.0
    kwargs["dataset"]["resolution"] = kwargs["model"]["resolution"]
    kwargs["dataset"]["gan_loss"] = kwargs["model"]["gan_loss"]
    kwargs["dataset"]["train_batch_size"] = kwargs["model"]["train_batch_size"]
    logger.info(kwargs)

    L.seed_everything(kwargs["seed"])

    # Initialize model
    model = CoMat(**kwargs["model"])
    dm = GanDataModule(**(kwargs["dataset"])) 

    wandb_logger = WandbLogger(project="CoMat")
    checkpoint_callbacks = ModelCheckpoint(
        dirpath='checkpoints/CoMat',
        filename='{step}-{g_loss:.4f}-{d_loss:.4f}',
        save_last=True,
        save_weights_only=True,
    )

    if gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = 'auto'

    trainer = L.Trainer(
        devices=gpus,
        num_nodes=nodes,
        accelerator='gpu',
        strategy=strategy,
        max_steps=kwargs["trainer"]["max_steps"],
        precision=kwargs["trainer"]["precision"],
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[checkpoint_callbacks],
    )
    trainer.fit(model, dm)
    logger.info(f"{checkpoint_callbacks.best_model_path} is the best model path")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CoMat model")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to use for training",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    main(**vars(args))
