import torchaudio
import librosa
import numpy as np
import soundfile as sf

from typing import Tuple


def audio_write(original_audio_path: str, target_audio_path: str, target_sr: int) -> None:
    signal, sample_rate = audio_check(original_audio_path, target_sr)
    sf.write(target_audio_path, signal, sample_rate)


def audio_check(audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    signal, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        signal = librosa.resample(signal.squeeze().numpy(), sr, target_sr)
    return signal, target_sr


def write_txt(text: str, path: str) -> None:
    with open(path, 'a') as f:
        f.write(text + "\n")
