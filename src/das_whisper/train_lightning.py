import os
from typing import Literal, Optional, Sequence

import defopt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .datautils import (
    VocalSegDataset,
    determine_default_config,
    get_audio_and_label_paths,
    get_cluster_codebook,
    load_data,
    slice_audios_and_labels,
    train_val_split,
)
from .model import load_model, save_model


class WhisperSegLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        *,
        learning_rate: float,
        weight_decay: float,
        lr_schedule: str,
        warmup_steps: int,
        total_training_steps: Optional[int],
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss.mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, logger=True, batch_size=batch["input_features"].size(0))
        self.log("train_loss_epoch", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch["input_features"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss.mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["input_features"].size(0), sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        if self.lr_schedule == "linear" and self.total_training_steps:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_training_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }

        return optimizer


class SavePretrainedCallback(Callback):
    def __init__(self, model_folder: str, tokenizer, max_to_keep: int):
        self.model_folder = model_folder
        self.tokenizer = tokenizer
        self.max_to_keep = max_to_keep

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        step = max(trainer.global_step, 0)
        save_model(pl_module.model, self.tokenizer, step, self.model_folder, self.max_to_keep)

    def on_train_end(self, trainer, pl_module):
        step = max(trainer.global_step, 0)
        save_model(pl_module.model, self.tokenizer, step, self.model_folder, self.max_to_keep)


def run(
    initial_model_path: str,
    model_folder: str,
    train_dataset_folder: str,
    *,
    n_device: int = 1,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    gpu_list: Optional[Sequence[int]] = None,
    print_every: int = 100,
    max_num_epochs: int = 3,
    max_num_iterations: Optional[int] = None,
    min_num_iterations: Optional[int] = 500,
    val_ratio: float = 0.0,
    max_length: int = 100,
    total_spec_columns: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 3e-6,
    lr_schedule: str = "linear",
    max_to_keep: int = -1,
    seed: Optional[int] = 66100,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    freeze_encoder: int = 0,
    dropout: float = 0.0,
    num_workers: int = 4,
    clear_cluster_codebook: int = 1,
    ignore_cluster: int = 0,
    validate_every: Optional[int] = None,
    validate_per_epoch: int = 0,
    save_every: Optional[int] = None,
    save_per_epoch: int = 0,
):
    """Train the segmenter using PyTorch Lightning."""
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    os.makedirs(model_folder, exist_ok=True)

    model, tokenizer = load_model(initial_model_path, total_spec_columns, dropout)

    if freeze_encoder:
        for para in model.model.encoder.parameters():
            para.requires_grad = False

    audio_path_list_train, label_path_list_train = get_audio_and_label_paths(train_dataset_folder)

    default_config = determine_default_config(audio_path_list_train, label_path_list_train, total_spec_columns, ignore_cluster=ignore_cluster)
    model.config.default_segmentation_config = default_config

    initial_codebook = {} if clear_cluster_codebook else getattr(model.config, "cluster_codebook", {})
    cluster_codebook = get_cluster_codebook(label_path_list_train, initial_codebook, ignore_cluster=ignore_cluster)
    model.config.cluster_codebook = cluster_codebook

    audio_list_train, label_list_train = load_data(
        audio_path_list_train,
        label_path_list_train,
        cluster_codebook=cluster_codebook,
        n_threads=20,
        default_config=default_config,
        ignore_cluster=ignore_cluster,
    )

    val_dataloader = None
    if val_ratio > 0:
        (audio_list_train, label_list_train), (audio_list_val, label_list_val) = train_val_split(audio_list_train, label_list_train, val_ratio)
        audio_list_val, label_list_val = slice_audios_and_labels(audio_list_val, label_list_val, total_spec_columns)
        val_dataset = VocalSegDataset(
            audio_list_val,
            label_list_val,
            tokenizer,
            max_length,
            total_spec_columns,
            model.config.species_codebook,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=num_workers > 0,
            pin_memory=False,
        )

    audio_list_train, label_list_train = slice_audios_and_labels(audio_list_train, label_list_train, total_spec_columns)

    training_dataset = VocalSegDataset(
        audio_list_train,
        label_list_train,
        tokenizer,
        max_length,
        total_spec_columns,
        model.config.species_codebook,
    )

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )

    if len(train_dataloader) == 0:
        raise RuntimeError("Too few examples (less than a batch) for training.")

    if max_num_iterations is not None and max_num_iterations > 0:
        max_num_epochs = int(np.ceil(max_num_iterations / len(train_dataloader)))
    else:
        assert max_num_epochs is not None and max_num_epochs > 0
        max_num_iterations = len(train_dataloader) * max_num_epochs
        if min_num_iterations is not None:
            max_num_iterations = max(max_num_iterations, min_num_iterations)
            max_num_epochs = int(np.ceil(max_num_iterations / len(train_dataloader)))

    lightning_module = WhisperSegLightningModule(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        total_training_steps=max_num_iterations,
    )

    callbacks = [
        SavePretrainedCallback(model_folder, tokenizer, max_to_keep),
        LearningRateMonitor(logging_interval="step"),
    ]

    monitor_metric = "val_loss" if val_dataloader is not None else "train_loss_epoch"
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder,
        filename="lightning-{epoch}-{step}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=max_to_keep if max_to_keep > 0 else -1,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    if val_dataloader is not None:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=3, mode="min"))

    devices = list(gpu_list) if gpu_list else n_device
    trainer_kwargs = dict(
        accelerator="auto" if device == "auto" else ("gpu" if device == "cuda" else device),
        devices=devices,
        max_epochs=max_num_epochs,
        max_steps=max_num_iterations,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    if val_dataloader is not None:
        if validate_every is not None:
            trainer_kwargs["val_check_interval"] = validate_every
        elif validate_per_epoch:
            trainer_kwargs["val_check_interval"] = 1.0

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    defopt.run(run)
