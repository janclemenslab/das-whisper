import os
from .model import WhisperSegmenter
import librosa
from tqdm import tqdm
import pandas as pd
from glob import glob
from typing import Optional, Sequence, Literal
import defopt
import torch
from torch.utils.data import DataLoader, Dataset


class AudioFolderDataset(Dataset):
    """Dataset that loads raw audio for each path in a folder."""

    def __init__(self, audio_paths):
        self.audio_paths = sorted(audio_paths)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio, sr = librosa.load(audio_path, sr=None)
        return {"audio": audio, "sr": sr, "path": audio_path}


def _single_item_collate(batch):
    """Unwrap batches since we use batch_size=1."""
    return batch[0]


def run(
    audio_folder: str,
    model_path: str,
    csv_save_path: str,
    *,
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
    device_ids: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    min_frequency: Optional[int] = None,
    spec_time_step: Optional[float] = None,
    num_trials: int = 1,
    num_workers: int = 0,
):
    """Segment audio files and export the results as CSV."""
    if device_ids is None:
        device_ids = [0]
    else:
        device_ids = list(device_ids)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    segmenter = WhisperSegmenter(model_path, device=device, device_ids=device_ids)

    audio_path_list = glob(audio_folder + "/*.wav") + glob(audio_folder + "/*.WAV")
    dataset = AudioFolderDataset(audio_path_list)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_single_item_collate,
    )
    overall_df = {"filename": [], "onset": [], "offset": [], "cluster": []}
    for sample in tqdm(dataloader):
        audio_path = sample["path"]
        audio_fname = os.path.basename(audio_path)
        audio = sample["audio"]
        sr = sample["sr"]
        res = segmenter.segment(
            audio,
            sr,
            min_frequency=min_frequency,
            spec_time_step=spec_time_step,
            num_trials=num_trials,
            batch_size=batch_size,
        )
        overall_df["filename"] += [audio_fname] * len(res["onset"])
        overall_df["onset"] += res["onset"]
        overall_df["offset"] += res["offset"]
        overall_df["cluster"] += res["cluster"]
    df = pd.DataFrame(overall_df)
    df.to_csv(csv_save_path, index=False)


if __name__ == "__main__":
    defopt.run(run)
