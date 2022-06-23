import glob
import os
import shutil
import hydra

from tqdm import tqdm


@hydra.main(config_path="configs", config_name="textgrids")
def main(cfg):
    textgrids = glob.glob(f"{cfg.target_textgrids_directory_path}/*.TextGrid")
    # copy all samples that are not in oovs to the new directory
    for wav in tqdm(textgrids):
        _, wav_filename = os.path.split(wav)
        filename = wav_filename[:-9]
        source_wav_path = f"{cfg.source_wav_directory_path}/{filename}.wav"
        source_txt_path = f"{cfg.source_wav_directory_path}/{filename}.txt"

        target_wav_path = f"{cfg.target_textgrids_directory_path}/{filename}.wav"
        target_txt_path = f"{cfg.target_textgrids_directory_path}/{filename}.txt"

        try:
            shutil.copyfile(source_wav_path, target_wav_path)
            shutil.copyfile(source_txt_path, target_txt_path)
        except FileNotFoundError:
            print(f"Couldn't find {source_wav_path} or {source_txt_path}")


if __name__ == "__main__":
    main()
