import glob
import os

import hydra
from tqdm import tqdm


@hydra.main(config_path="configs", config_name="write_nonaligned")
def main(cfg):
    wavs = glob.glob(f"{cfg.wav_directory_path}/*.wav")
    textgrids = glob.glob(f"{cfg.textgrids_directory_path}/*.TextGrid")

    textgrids_filenames = []
    for t in textgrids:
        _, t_filename = os.path.split(t)
        textgrids_filenames.append(t_filename[:-9])

    for wav in tqdm(wavs):
        _, wav_filename = os.path.split(wav)
        if wav_filename[:-4] not in textgrids_filenames:
            print(wav_filename)


if __name__ == "__main__":
    main()
