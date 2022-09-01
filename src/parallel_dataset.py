import glob
import shutil
import os
import hydra

from typing import Dict, List
from tqdm import tqdm


def copy_wavs(cfg: Dict, speaker: int, emotion: str, emotion_id: int, part: str):
    """
    Function maps original filename into {speaker_id}_{file_id}_{emo_id} and copies into 1 directory along with
    similar-named txt file with audio transcription.
    :param cfg: config hydra dict
    :param speaker: str, original name of a directory containing sub-directories with this speaker audios
    :param emotion: str, name of emotion
    :param emotion_id: int, mapping of string emotion into id
    :param part: str, train, evaluation or test
    :return: List of strings with new filenames (without extensions)
    """
    val_ids_list = []
    filenames = os.listdir(f"{cfg.source_data_directory}/{speaker}/{emotion}/{part}")
    for filename in filenames:
        path = f"{cfg.source_data_directory}/{speaker}/{emotion}/{part}/{filename}"
        speaker_id, file_id = filename[:-4].split("_")
        speaker_id = int(speaker_id)
        file_id = int(file_id)
        new_path = f"{cfg.target_directory_path}/wavs/{speaker_id}_{file_id}_{emotion_id}.wav"
        txt_path = f"{cfg.target_directory_path}/wavs/{speaker_id}_{file_id}_{emotion_id}.txt"
        if os.path.exists(path) and os.path.exists(txt_path):
            shutil.copyfile(path, new_path)
            if part == "test":
                val_ids_list.append(f"{speaker_id}_{file_id}_{emotion_id}")
    return val_ids_list


@hydra.main(config_path="configs", config_name="parallel_dataset")
def build_dataset(cfg):
    os.makedirs(cfg.target_directory_path, exist_ok=True)
    os.makedirs(f"{cfg.target_directory_path}/wavs", exist_ok=True)

    speakers = os.listdir(cfg.source_data_directory)[10:21]
    speaker_encodings = cfg.speaker_encodings
    emotions_dict = dict(zip(cfg.emotions, cfg.emotion_ids))

    all_val_ids = []

    for i, speaker in tqdm(enumerate(speakers)):
        if os.path.isdir(f"{cfg.source_data_directory}/{speaker}"):
            speaker_transcripts = open(f"{cfg.source_data_directory}/{speaker}/{speaker}.txt",
                                       encoding=speaker_encodings[i]).readlines()

            for speaker_txt in speaker_transcripts:
                try:
                    filename, text, emotion = speaker_txt.split("\t")
                    speaker_id, file_id = filename.split("_")
                    emotion = emotion[:-1]
                    if emotion[-1] == " ":
                        emotion = emotion[:-1]
                    emotion_id = emotions_dict[emotion]
                    speaker_id = int(speaker_id)
                    file_id = int(file_id)
                    with open(f"{cfg.target_directory_path}/wavs/{speaker_id}_{file_id}_{emotion_id}.txt", "w") as f:
                        f.write(text)
                except ValueError:
                    pass

            for emotion in cfg.emotions:
                emotion_id = emotions_dict[emotion]
                for part in ["train", "evaluation", "test"]:
                    res = copy_wavs(cfg, speaker, emotion, emotion_id, part)
                    if len(res) > 0:
                        all_val_ids.extend(res)

    with open(f"{cfg.target_directory_path}/val_ids.txt", "w") as f:
        f.write("|".join(all_val_ids))

    print(f"Saved wavs and txts in {cfg.target_directory_path} folder!")


if __name__ == "__main__":
    build_dataset()
