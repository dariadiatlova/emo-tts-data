import glob
import os
import shutil
import hydra

from tqdm import tqdm


@hydra.main(config_path="configs", config_name="filter_oovs")
def main(cfg):
    filename = cfg.oovs_file_path
    oovs_filenames = []
    # collect oovs filenames
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            name = line.split(" ")[0][1:-1]
            oovs_filenames.append(name)

    original_wavs = glob.glob(f"{cfg.source_directory_path}/*.wav")
    # copy all samples that are not in oovs to the new directory
    for wav in tqdm(original_wavs):
        _, wav_filename = os.path.split(wav)
        filename = wav_filename[:-5]
        if filename not in oovs_filenames:
            source_wav_path = f"{cfg.source_directory_path}/{filename}.wav"
            source_txt_path = f"{cfg.source_directory_path}/{filename}.txt"

            target_wav_path = f"{cfg.target_directory_path}/{filename}.wav"
            target_txt_path = f"{cfg.target_directory_path}/{filename}.txt"

            try:
                shutil.copyfile(source_wav_path, target_wav_path)
                shutil.copyfile(source_txt_path, target_txt_path)
            except FileNotFoundError:
                print(f"Couldn't find {source_wav_path} or {source_txt_path}")


if __name__ == "__main__":
    main()
