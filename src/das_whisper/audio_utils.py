import pandas as pd
from transformers import WhisperFeatureExtractor
from transformers.audio_utils import mel_filter_bank
import numpy as np


def get_n_fft_given_sr(sr):
    if sr <= 32000:
        n_fft = 512
    elif sr <= 80000:
        n_fft = 1024
    elif sr <= 150000:
        n_fft = 2048
    elif sr <= 300000:
        n_fft = 4096
    else:
        n_fft = 8192
    return n_fft


class WhisperSegFeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, sr, spec_time_step, min_frequency=None, max_frequency=None, chunk_length=30):
        hop_length = int(spec_time_step * sr)
        # if hop_length != spec_time_step * sr:
        #     print("Warning: spec_time_step * sr must be an integer. Consider changing the sampling rate sr.")

        n_fft = get_n_fft_given_sr(sr)

        if min_frequency is None:
            min_frequency = 0
        if max_frequency is None:
            max_frequency = sr // 2

        super().__init__(
            feature_size=80,
            sampling_rate=sr,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=0.0,
            return_attention_mask=False,
        )

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=80,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            sampling_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )


def slice_audio_and_label(audio, label, sr, start_time, end_time):
    sliced_audio = audio[int(start_time * sr) : int(end_time * sr)]
    duration = len(sliced_audio) / sr
    ## get the actual ending time
    end_time = start_time + duration

    onsets = np.array(label["onset"])
    offsets = np.array(label["offset"])
    clusters = list(label["cluster"])

    target_indices = np.argwhere(np.logical_and(onsets < end_time, offsets > start_time))[:, 0]

    sliced_onsets = [max(0, onsets[idx] - start_time) for idx in target_indices]
    sliced_offsets = [min(offsets[idx] - start_time, end_time - start_time) for idx in target_indices]
    sliced_clusters = [clusters[idx] for idx in target_indices]

    sliced_label = {
        "onset": sliced_onsets,
        "offset": sliced_offsets,
        "cluster": sliced_clusters,
    }

    if isinstance(label, pd.DataFrame):
        sliced_label = pd.DataFrame(sliced_label)

    return sliced_audio, sliced_label
