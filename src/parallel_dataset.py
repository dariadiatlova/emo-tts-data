import glob
import shutil
import os
import hydra

from typing import Dict
from tqdm import tqdm


def get_speaker_transcripts(txt_path: str) -> Dict:
    """
    :param txt_path: path to the file that locates in each of the speakers' folders and contains
    audio id, transcription and emotion
    :return: Dict, looks like: {"0011_000001.wav": "The nine the eggs, I keep.",
                                "0011_000002.wav": "I did go, and made many prisoners."}
    """
    result_dict = {}
    with open(txt_path) as f:
        lines = f.readlines()
        for line in lines:
            string = line.split("\t")
            result_dict[f"{string[0]}.wav"] = line[1]
    return result_dict


def write_txt(text: str, path: str) -> None:
    with open(path, 'w') as f:
        f.write(text + "\n")


@hydra.main(config_path="configs", config_name="parallel_dataset")
def build_dataset(cfg):
    manifest_path = cfg.target_directory_path + "/parallel_manifest"
    wavs_path = cfg.target_directory_path + "/wavs"
    emotion_dict = dict(zip(cfg.emotions, cfg.emotion_ids))
    for speaker_id in tqdm(cfg.original_speaker_ids):
        # print(cfg.original_speaker_ids)
        # print(speaker_id)
        speaker_transcripts_dict = get_speaker_transcripts(f"{cfg.source_data_directory}/{speaker_id}/{speaker_id}.txt")
        # each speaker has 5 folders for emotions: "Neutral", "Angry", "Happy", "Sad", "Surprise"
        for emotion in cfg.emotions:
            # each folder of emotion has a division on tran/evaluation/test in proportion 85% / 9% / 6%
            for part in ["train", "evaluation", "test"]:
                directory = f"{cfg.source_data_directory}/{speaker_id}/{emotion}/{part}"
                # write to wavs all filenames in train, evaluation or test directory
                wavs = glob.glob(f"{directory}/*.wav")
                # create 3 Manifests files for train, evaluation and test
                manifest_filename = f"{manifest_path}_{part}.txt"
                for audio_id, wav in enumerate(wavs):
                    emotion_id = emotion_dict[emotion]
                    new_absolute_wav_path = f"{wavs_path}/{speaker_id}_{audio_id}_{emotion_id}.wav"
                    new_relative_wav_path = f"vk_etts_data/wavs/{speaker_id}_{audio_id}_{emotion_id}.wav"
                    shutil.copyfile(wav, new_absolute_wav_path)
                    _, wav_filename = os.path.split(wav)
                    new_txt_path = f"{wavs_path}/{speaker_id}_{audio_id}_{emotion_id}.txt"
                    transcription = speaker_transcripts_dict[wav_filename]
                    # write transcription, file name of txt == file name of wav for future TextGrids generation
                    write_txt(transcription, new_txt_path)
                    # write data to Manifest: "/path/to/audio.wav"|"speaker_id"|"emotion_id"|"text"
                    write_txt(f"{new_relative_wav_path}|{speaker_id}|{emotion_id}|{transcription}", manifest_filename)
    print(f"Saved wavs and manifests in {cfg.target_directory_path} folder!")


if __name__ == "__main__":
    build_dataset()
