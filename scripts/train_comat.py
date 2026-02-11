"""Train CoMat from a JSONL file containing a ``text`` or ``prompt`` field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


class PromptDataset:
    def __init__(self, path: Path):
        self.rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text") or row.get("prompt")
                if text:
                    self.rows.append({**row, "text": text})
        if not self.rows:
            raise ValueError(f"No prompts found in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model_config", type=Path, default=Path("configs/comat/model/sd21.yaml"))
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="checkpoints/comat")
    parser.add_argument("--precision", default="bf16-mixed")
    args = parser.parse_args()

    from lightning import Trainer, seed_everything
    from torch.utils.data import DataLoader
    from models.comat import CoMat

    seed_everything(995, workers=True)
    config = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
    model = CoMat(**config)
    loader = DataLoader(
        PromptDataset(args.prompts),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    trainer = Trainer(
        max_steps=args.max_steps,
        precision=args.precision,
        default_root_dir=args.output_dir,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
