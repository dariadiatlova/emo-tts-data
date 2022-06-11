import torchaudio
import librosa
import numpy as np
import soundfile as sf

from typing import Tuple


def audio_write(audio_path: str, target_sr: int) -> None:
    signal, sample_rate = audio_check(audio_path, target_sr)
    sf.write(audio_path, signal, sample_rate)


def audio_check(audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    signal, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        signal = librosa.resample(signal, sr, target_sr)
    return signal, target_sr


def write_txt(text: str, path: str) -> None:
    with open(path, 'a') as f:
        f.write(text + "\n")
