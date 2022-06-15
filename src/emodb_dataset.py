import glob
import os
from typing import Dict, Tuple

import hydra
import numpy as np
from tqdm import tqdm

from utils import write_txt, audio_write


def get_transcriptions(txt_path: str, encoding_type: str = "us-ascii") -> Dict:
    """
    :param txt_path: path to the file that locates in each of the speakers' folders and contains
    audio id, transcription and emotion
    :param encoding_type: txt encoding type: [utf-16le, us-ascii, iso-8859-1]
    :return: Dict, looks like: {"0001": "Author of the danger trail, Philip Steels, etc.",
                                "0011_000002.wav": "I did go, and made many prisoners."}
    """
    result_dict = {}
    with open(txt_path, "r", encoding=encoding_type) as f:
        lines = f.readlines()
        for line in lines:
            if line[9] == "a":
                code = line[10:14]
                sentence = line[15:-3]
                result_dict[code] = sentence
    return result_dict


def get_speaker_wavs(wavs_dir_path: str) -> Tuple[Dict, Dict, Dict]:
    train_dict = {}
    val_dict = {}
    test_dict = {}
    paths = glob.glob(f"{wavs_dir_path}/*.wav")
    # 85%, 9%, 6% – train test split
    train_size = int(len(paths) * 0.85)
    val_size = int(len(paths) * 0.09)
    np.random.shuffle(paths)
    for i, path in enumerate(paths):
        _, wav_filename = os.path.split(path)
        code = wav_filename[-8:-4]
        if i < train_size:
            train_dict[code] = path
        elif i < train_size + val_size:
            val_dict[code] = path
        else:
            test_dict[code] = path
    return train_dict, val_dict, test_dict


@hydra.main(config_path="configs", config_name="emodb_dataset")
def build_dataset(cfg):
    manifest_path = cfg.target_directory_path + "/emodb_manifest"
    wavs_path = cfg.target_directory_path + "/wavs"
    emotion_dict = dict(zip(cfg.emotions, cfg.emotion_ids))
    target_speaker_id_dict = dict(zip(cfg.original_speaker_names, cfg.target_speaker_ids))
    transcripts_dictionary = get_transcriptions(f"{cfg.source_data_directory}/cmuarctic.data")
    for name in tqdm(cfg.original_speaker_names):
        for emotion in cfg.emotions:
            audio_id = 1
            emotion_id = emotion_dict[emotion]
            speaker_id = target_speaker_id_dict[name]
            folder_path = f"{cfg.source_data_directory}/{name}_{emotion}"
            train_dict, val_dict, test_dict = get_speaker_wavs(folder_path)
            # each folder of emotion has a division on tran/evaluation/test in proportion 85% / 9% / 6%
            for d, part in zip([train_dict, val_dict, test_dict], ["train", "evaluation", "test"]):
                for code, original_wav_path in zip(d.keys(), d.values()):
                    transcription = transcripts_dictionary[code]
                    target_wav_path = f"{cfg.target_directory_path}/wavs/{speaker_id}_{audio_id}_{emotion_id}.wav"
                    audio_write(original_wav_path, target_wav_path, cfg.target_sample_rate)
                    manifest_filename = f"{manifest_path}_{part}.txt"
                    new_txt_path = f"{cfg.target_directory_path}/wavs/{speaker_id}_{audio_id}_{emotion_id}.txt"
                    write_txt(transcription, new_txt_path)
                    # write data to Manifest: "/path/to/audio.wav"|"speaker_id"|"emotion_id"|"text"
                    write_txt(f"{target_wav_path}|{speaker_id}|{emotion_id}|{transcription}", manifest_filename)
    print(f"Saved wavs and manifests in {cfg.target_directory_path} folder!")


if __name__ == "__main__":
    build_dataset()
