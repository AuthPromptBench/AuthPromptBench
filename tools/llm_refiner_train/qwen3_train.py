from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset.clear_prompts_dataset import ClearPromptsDataset
from models.utils.logger import get_logger
from models.components.refiner.llmrefiner import Qwen3LLMRefiner
import os
import wandb
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from huggingface_hub import login
from typing import List

login(token="")


logger = get_logger(__name__)
L.seed_everything(995)

wandb.login(key="")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def main(
        gpus: int,
        nodes: int,
        dataset_paths: List[str],
        name: str,
        version: str
):
    model_name = "Qwen/Qwen3-8B"
    batch_size = 2
    model = Qwen3LLMRefiner(model_name=model_name)

    train_dataset = ClearPromptsDataset(dataset_paths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    wandb_logger = WandbLogger(project="llm-refiner", name=name)
    if not os.path.exists(os.path.join(PROJECT_DIR, "checkpoints", name)):
        os.makedirs(os.path.join(PROJECT_DIR, "checkpoints", name, f"version_{version}"))
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(PROJECT_DIR, "checkpoints", name, f"version_{version}"),
        filename="refiner-{epoch:02d}-{train_loss:.2f}",
        mode="min",
        every_n_train_steps=25,
        save_top_k=3
    )

    if gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = 'auto'

    trainer = Trainer(
        devices=gpus,
        num_nodes=nodes,
        accelerator="gpu",
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=3,
        accumulate_grad_batches=32,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader)
    logger.info(f"{checkpoint_callback.best_model_path} is the best model path")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Mistral LoRA model")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--dataset_paths", type=str, nargs='+', help="Paths to the datasets", default=[
        'data/diffusionDB/part-000001/sft_01.jsonl',
        'data/diffusionDB/part-000002/sft_02.jsonl',
        'data/diffusionDB/part-000003/sft_03.jsonl',
        'data/diffusionDB/part-000004/sft_04.jsonl',
        'data/diffusionDB/part-000005/sft_05.jsonl',
    ])
    parser.add_argument("--name", type=str, help="Name of the experiment")
    parser.add_argument("--version", type=str, help="Version of the model")
    args = parser.parse_args()

    main(gpus=args.gpus, nodes=args.nodes, dataset_paths=args.dataset_paths, name=args.name, version=args.version)