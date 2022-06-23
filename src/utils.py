import torchaudio
import librosa
import numpy as np
import soundfile as sf

from num2words import num2words
from typing import Tuple, Optional


def check_for_numbers(text: str, mode: str = "replace") -> Optional[bool, str]:
    transcription = text.split(" ")

    for i, char in enumerate(transcription):
        # return transcription if no integers found or open integers into words
        if mode == "replace":
            try:
                _int = int(char)
                transcription[i] = num2words(_int, lang='en')
            except ValueError:
                pass

        # return true if it's impossible to convert any character into string
        else:
            try:
                int(char)
                return False
            except ValueError:
                pass
    return " ".join(transcription) if mode == "replace" else True


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
