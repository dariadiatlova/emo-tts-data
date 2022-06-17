import glob
import os
import torchaudio
import hydra
import numpy as np

from collections import defaultdict
from tqdm import tqdm


@hydra.main(config_path="configs", config_name="profiler")
def main(cfg):
    all_wav_paths = glob.glob(f"{cfg.wavs_directory_path}/*.wav")
    emotion_dict = dict(zip(cfg.emotion_ids, cfg.emotions))
    cfg_profiler_emotion_dict = defaultdict(lambda: 0)
    for wav_path in tqdm(all_wav_paths):
        _, wav_filename = os.path.split(wav_path)
        sr, wav = torchaudio.load(wav_path)
        seconds_len = wav.squeeze() // sr
        emotion = emotion_dict[int(wav_filename[-5])]
        cfg_profiler_emotion_dict[emotion] += seconds_len
    total_dataset_len = 0
    for emotion, sec_len in zip(list(cfg_profiler_emotion_dict.keys()), list(cfg_profiler_emotion_dict.values())):
        hours = np.round(sec_len / 60 / 60, 3)
        print(f"Emotion '{emotion}': {hours} hours.")
        total_dataset_len += hours
    print(f"Total dataset size: {total_dataset_len} hours.")


if __name__ == "__main__":
    main()
