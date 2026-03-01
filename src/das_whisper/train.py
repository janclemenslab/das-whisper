import os, sys, inspect
from typing import Optional, Sequence
from tqdm import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import shutil
import json
import defopt
from .utils import *
from .model import *
from .datautils import *
from .evaluate import evaluate
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import Literal


def run(
    initial_model_path: str,
    model_folder: str,
    train_dataset_folder: str,
    *,
    n_device: int = 1,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    gpu_list: Optional[Sequence[int]] = None,
    print_every: int = 100,
    validate_every: Optional[int] = None,
    validate_per_epoch: int = 0,
    save_every: Optional[int] = None,
    save_per_epoch: int = 0,
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
):
    """Train a whisperseg model

    The argument `initial_model_path` can point to a local folder or to a pretrained model on huggingface:
    - openai/whisper-large
    - openai/whisper-medium
    - openai/whisper-small
    - openai/whisper-base
    - openai/whisper-tiny
    - nccratliri/whisperseg-large-ms
    - nccratliri/whisperseg-animal-vad
    - nccratliri/whisperseg-base-animal-vad
    - nccratliri/whisperseg-canary
    - nccratliri/whisper-large

    The argument `train_dataset_folder` can point to a local folder or to a dataset on huggingface:
    - nccratliri/bengalese-finch-subset-with-csv-label
    - nccratliri/vad-animals
    - nccratliri/whisperseg-conda-env
    - nccratliri/vad-multi-species
    - nccratliri/vad-zebra-finch
    - nccratliri/vad-bengalese-finch
    - nccratliri/vad-marmoset
    - nccratliri/vad-mouse
    - nccratliri/vad-human-ava-speech
    """
    if seed is not None:
        np.random.seed(seed)

    if val_ratio == 0.0:
        validate_every = None
        validate_per_epoch = None

    os.makedirs(model_folder, exist_ok=True)

    if gpu_list is None:
        gpu_list = np.arange(n_device).tolist()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model, tokenizer = load_model(initial_model_path, total_spec_columns, dropout)

    model = model.to(device)

    if freeze_encoder:
        for para in model.model.encoder.parameters():
            para.requires_grad = False
    else:
        for para in model.model.encoder.parameters():
            para.requires_grad = True

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    model = nn.DataParallel(model, gpu_list)

    segmenter = WhisperSegmenterForEval(model=model, tokenizer=tokenizer)

    if clear_cluster_codebook:
        segmenter.update_cluster_codebook({})

    # scaler = torch.cuda.amp.GradScaler()
    print(train_dataset_folder)
    train_dataset_folder = download_data(train_dataset_folder, local_data_path="datasets")

    audio_path_list_train, label_path_list_train = get_audio_and_label_paths(train_dataset_folder)

    default_config = determine_default_config(
        audio_path_list_train, label_path_list_train, total_spec_columns, ignore_cluster=ignore_cluster
    )
    ## store the default segmentation config
    segmenter.model.config.default_segmentation_config = default_config
    segmenter.default_segmentation_config = default_config

    cluster_codebook = get_cluster_codebook(label_path_list_train, segmenter.cluster_codebook, ignore_cluster=ignore_cluster)
    segmenter.update_cluster_codebook(cluster_codebook)

    audio_list_train, label_list_train = load_data(
        audio_path_list_train,
        label_path_list_train,
        cluster_codebook=cluster_codebook,
        n_threads=20,
        default_config=default_config,
        ignore_cluster=ignore_cluster,
    )

    if val_ratio > 0:
        (audio_list_train, label_list_train), (audio_list_val, label_list_val) = train_val_split(
            audio_list_train, label_list_train, val_ratio
        )

    audio_list_train, label_list_train = slice_audios_and_labels(audio_list_train, label_list_train, total_spec_columns)

    training_dataset = VocalSegDataset(
        audio_list_train,
        label_list_train,
        tokenizer,
        max_length,
        total_spec_columns,
        model.module.config.species_codebook,
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True,
        pin_memory=False,
    )

    # if the training dataset is really too small, then trigger the error
    if len(training_dataloader) == 0:
        print("Error: Too few examples (less than a batch) for training! Exit!")
        sys.exit(1)

    if max_num_iterations is not None and max_num_iterations > 0:
        max_num_epochs = int(np.ceil(max_num_iterations / len(training_dataloader)))
    else:
        assert max_num_epochs is not None and max_num_epochs > 0
        max_num_iterations = len(training_dataloader) * max_num_epochs

        if min_num_iterations is not None:
            max_num_iterations = max(max_num_iterations, min_num_iterations)
            max_num_epochs = int(np.ceil(max_num_iterations / len(training_dataloader)))

    if lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_num_iterations
        )
    else:
        scheduler = None

    model.train()
    training_loss_value_list = []
    val_score_history = []
    eary_stop = False
    current_step = 0

    progress = 0
    eta = None
    start_time = time.time()

    for epoch in range(max_num_epochs + 1):  # This +1 is to ensure current_step can reach max_num_iterations
        training_dataloader.epoch = epoch
        for count, batch in enumerate(tqdm(training_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            model_out = model(**batch)
            loss = model_out.loss.mean()
            loss.backward()
            optimizer.step()

            training_loss_value_list.append(loss.item())

            if scheduler is not None:
                scheduler.step()

            current_step += 1

            current_time = time.time()
            current_progress = int(np.round(current_step / max_num_iterations * 100))
            eta = int(
                (current_time - start_time) / (current_step / max_num_iterations) * (1 - current_step / max_num_iterations)
            )
            eta_hours = eta // 3600
            eta_minutes = (eta % 3600) // 60
            eta_seconds = (eta % 3600) % 60
            if current_progress > progress:
                json.dump(
                    {"progress": current_progress, "eta": "%02d:%02d:%02d" % (eta_hours, eta_minutes, eta_seconds)},
                    open(model_folder + "/status.json", "w"),
                )
            progress = current_progress

            if current_step % print_every == 0:
                print(
                    "Epoch: %d, current_step: %d, learning rate: %f, Loss: %.4f"
                    % (epoch, current_step, get_lr(optimizer)[0], np.mean(training_loss_value_list))
                )
                training_loss_value_list = []

            if (validate_every is not None and current_step % validate_every == 0) or (
                validate_per_epoch and count == len(training_dataloader) - 1
            ):
                print("Start validation ...")
                model.eval()
                ## in the validation set, set the num_trails to 1
                eval_res = evaluate(
                    audio_list_val,
                    label_list_val,
                    segmenter,
                    batch_size,
                    max_length,
                    num_trials=1,
                    num_beams=1,
                    target_cluster=None,
                )

                print(
                    "Epoch: %d, current_step: %d, validation segment F1 score: %.2f, frame F1 score: %.2f"
                    % (epoch, current_step, eval_res["segment_wise"][-1], eval_res["frame_wise"][-1])
                )
                val_score_history.append((current_step, (eval_res["segment_wise"][-1] + eval_res["frame_wise"][-1]) * 0.5))

                model.train()

            if (save_every is not None and current_step % save_every == 0) or (
                save_per_epoch and count == len(training_dataloader) - 1
            ):
                model.eval()
                save_model(model, tokenizer, current_step, model_folder, max_to_keep)
                model.train()

            if current_step >= 0.5 * max_num_iterations:  ## training has been half-way done
                ## validation score keep decreasing for 2 validation steps
                if (
                    len(val_score_history) >= 3
                    and val_score_history[-1][1] < val_score_history[-2][1]
                    and val_score_history[-2][1] < val_score_history[-3][1]
                ):
                    eary_stop = True

            if current_step >= max_num_iterations or eary_stop:
                if not os.path.exists(model_folder + "/checkpoint-%d" % (current_step)):
                    model.eval()
                    save_model(model, tokenizer, current_step, model_folder, max_to_keep)
                break

        if current_step >= max_num_iterations or eary_stop:
            break

    json.dump({"progress": 100, "eta": "%02d:%02d:%02d" % (0, 0, 0)}, open(model_folder + "/status.json", "w"))

    best_checkpoint_batch_number = None
    if len(val_score_history) > 0:
        best_checkpoint_batch_number = sorted(val_score_history, key=lambda x: -x[1])[0][0]
    else:
        ckpt_list = glob(model_folder + "/checkpoint-*")
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            ckpt_name = ckpt_list[-1]
            best_checkpoint_batch_number = int(ckpt_name.split("-")[-1])

    if best_checkpoint_batch_number is not None:
        print(
            "The best checkpoint on validation set is: %s," % (model_folder + "/checkpoint-%d" % (best_checkpoint_batch_number))
        )
        os.system(
            "cp -r %s %s"
            % (model_folder + "/checkpoint-%d" % (best_checkpoint_batch_number), model_folder + "/final_checkpoint")
        )
        # remove other checkpoints
        os.system("rm -r %s" % (model_folder + "/checkpoint-*"))
    try:
        os.remove(model_folder + "/status.json")
    except:
        pass

    print("All Done!")


if __name__ == "__main__":
    defopt.run(run)
