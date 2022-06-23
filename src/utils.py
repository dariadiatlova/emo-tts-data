import torchaudio
import librosa
import numpy as np
import soundfile as sf

from typing import Tuple


def check_for_numbers(text: str) -> bool:
    transcription = text.split(" ")
    # return true if it's impossible to convert any character into string
    for char in transcription:
        try:
            int(char)
            return False
        except ValueError:
            pass
    return True


def audio_write(original_audio_path: str, target_audio_path: str, target_sr: int) -> None:
    signal, sample_rate = audio_check(original_audio_path, target_sr)
    sf.write(target_audio_path, signal, sample_rate)


def audio_check(audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    signal, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        signal = librosa.resample(signal.squeeze().numpy(), sr, target_sr)
    else:
        signal = signal.squeeze().numpy()
    return signal, target_sr


def write_txt(text: str, path: str) -> None:
    with open(path, 'a') as f:
        f.write(text + "\n")
