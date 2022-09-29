import glob
import shutil
import os
import hydra

from typing import Dict, List
from tqdm import tqdm


def copy_wavs(cfg, speaker, emotion, part, speaker_emotions_dict, speaker_txt_dict):
    val_ids_list = []
    test_ids_list = []
    filenames = os.listdir(f"{cfg.source_data_directory}/{speaker}/{emotion}/{part}")
    for i, filename in enumerate(filenames):
        path = f"{cfg.source_data_directory}/{speaker}/{emotion}/{part}/{filename}"
        speaker_id, file_id = filename[:-4].split("_")
        try:
            emotion_id = speaker_emotions_dict[filename[:-4]]
            speaker_id = int(speaker_id)
            new_path = f"{cfg.target_directory_path}/wavs/{speaker_id}_{i}_{emotion_id}.wav"
            txt_path = f"{cfg.target_directory_path}/wavs/{speaker_id}_{i}_{emotion_id}.txt"
            if os.path.exists(path):
                shutil.copyfile(path, new_path)
                with open(txt_path, "w") as f:
                    f.write(speaker_txt_dict[filename[:-4]])
                if part == "evaluation":
                    val_ids_list.append(f"{speaker_id}_{i}_{emotion_id}")
                elif part == "test":
                    test_ids_list.append(f"{speaker_id}_{i}_{emotion_id}")
        except KeyError:
            pass
    return val_ids_list, test_ids_list


@hydra.main(config_path="configs", config_name="parallel_dataset")
def build_dataset(cfg):
    os.makedirs(cfg.target_directory_path, exist_ok=True)
    os.makedirs(f"{cfg.target_directory_path}/wavs", exist_ok=True)

    speakers = os.listdir(cfg.source_data_directory)[10:21]
    speaker_encodings = cfg.speaker_encodings
    emotions_dict = dict(zip(cfg.emotions, cfg.emotion_ids))

    all_val_ids = []
    all_test_ids = []

    for i, speaker in tqdm(enumerate(speakers)):
        if os.path.isdir(f"{cfg.source_data_directory}/{speaker}"):
            speaker_transcripts = open(f"{cfg.source_data_directory}/{speaker}/{speaker}.txt",
                                       encoding=speaker_encodings[i]).readlines()

            speaker_filename_ids = []
            speaker_texts = []
            speaker_emotions = []
            for i, speaker_txt in enumerate(speaker_transcripts):
                try:
                    filename, text, emotion = speaker_txt.split("\t")
                    if len(emotion) > 1:
                        speaker_emotions.append(emotions_dict[emotion[:-1]])
                        speaker_filename_ids.append(filename)
                        speaker_texts.append(text)
                except (KeyError, ValueError) as e:
                    pass

            speaker_emotions_dict = dict(zip(speaker_filename_ids, speaker_emotions))
            speaker_txt_dict = dict(zip(speaker_filename_ids, speaker_texts))

            for emotion in cfg.emotions:
                for part in ["train", "evaluation", "test"]:
                    val_ids, test_ids = copy_wavs(cfg, speaker, emotion, part, speaker_emotions_dict, speaker_txt_dict)
                    if len(val_ids) > 0:
                        all_val_ids.extend(val_ids)
                    if len(test_ids) > 0:
                        all_test_ids.extend(test_ids)

    with open(f"{cfg.target_directory_path}/val_ids.txt", "w") as f:
        f.write("|".join(all_val_ids))

    with open(f"{cfg.target_directory_path}/test_ids.txt", "w") as f:
        f.write("|".join(all_test_ids))

    print(f"Saved wavs and txts in {cfg.target_directory_path} folder!")


if __name__ == "__main__":
    build_dataset()
