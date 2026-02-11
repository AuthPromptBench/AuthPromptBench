import torch
from torch import nn
from lightning.pytorch import LightningModule
from transformers import CLIPTokenizer, CLIPTextModel
from peft import get_peft_model, LoraConfig
from diffusers.optimization import get_scheduler

class NoxEye(LightningModule):
    def __init__(
        self,
        pretrain_model: str = "stabilityai/stable-diffusion-2-1",
        num_labels: int = 2,
        use_lora: bool = True,
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0.1,
        dtype: torch.dtype = torch.float16,
        lr: float = 5e-5,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(pretrain_model, subfolder="text_encoder")
        self.text_encoder.to(dtype)
        self.hidden_size = self.text_encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels, dtype=torch.float32)
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        else:
            self.text_encoder.requires_grad_(True)

    @torch.amp.autocast(device_type="cuda")
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = text_outputs.last_hidden_state #(batch_size, seq_len, hidden_size)
        logits = self.classifier(hidden_state) #(batch_size, seq_len, num_labels)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['adv_text_ids'], batch['adv_text_attention_mask'])
        loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), batch['label'].view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['adv_text_ids'], batch['adv_text_attention_mask'])
        loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, logits.size(-1)), batch['label'].view(-1))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        logits = self(batch['adv_text_ids'], batch['adv_text_attention_mask'])
        predictions = torch.softmax(logits, dim=-1)
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def state_dict(self):
        # Only keep trainable parameters in the state dict
        state = super().state_dict()
        return {k: v for k, v in state.items() if v.requires_grad}
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint into the model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Checkpoint loaded from {checkpoint_path}")
    



