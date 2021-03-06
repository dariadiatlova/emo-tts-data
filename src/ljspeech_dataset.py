import os
import shutil

import hydra
import pandas as pd
from tqdm import tqdm

from utils import write_txt, check_for_numbers


@hydra.main(config_path="configs", config_name="ljspeech_dataset")
def main(cfg):
    csv_path = f"{cfg.source_directory_path}/metadata.csv"
    split_directories = [f"{cfg.source_directory_path}/split/{txt_filename}" for txt_filename in cfg.split_filenames]
    wav_directory_path = f"{cfg.source_directory_path}/wavs"
    target_directory_path = cfg.target_directory_path
    df = pd.read_csv(csv_path, sep="|")
    transcriptions_dictionary = dict(zip(list(df.iloc[:, 0]), list(df.iloc[:, 1])))
    manifests_array = [f"{cfg.source_directory_path}/ljspeech_manifest_{i}" for i in cfg.split_filenames]

    # iterate through train/val/test files
    for i, txt_filename in enumerate(split_directories):
        manifest_file_path = manifests_array[i]
        with open(txt_filename, encoding="us-ascii") as f:
            lines = f.readlines()
            for path in tqdm(lines):
                _, wav_filename = os.path.split(path)
                filename = wav_filename[:-5]
                if not cfg.use_unalignable or filename not in cfg.unalignable_filenames:
                    try:
                        new_wav_path = f"{target_directory_path}/{filename}.wav"
                        new_txt_path = f"{target_directory_path}/{filename}.txt"
                        transcription = transcriptions_dictionary[filename]
                        if cfg.use_non_numbers:
                            if cfg.mode == "replace":
                                transcription = check_for_numbers(transcription, cfg.mode)
                            else:
                                if not check_for_numbers(transcription, cfg.mode):
                                    break
                        shutil.copyfile(f"{wav_directory_path}/{filename}.wav", new_wav_path)
                        write_txt(transcription, new_txt_path)
                        write_txt(f"{new_wav_path}|{transcription}", manifest_file_path)
                    except KeyError:
                        print(f"Couldn't find a transcription for {filename} :(")


if __name__ == "__main__":
    main()
