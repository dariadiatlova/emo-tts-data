import glob
import shutil
import os
import hydra

from typing import Dict
from tqdm import tqdm


def get_speaker_transcripts(txt_path: str, encoding_type: str) -> Dict:
    """
    :param txt_path: path to the file that locates in each of the speakers' folders and contains
    audio id, transcription and emotion
    :param encoding_type: txt encoding type: [utf-16le, us-ascii, iso-8859-1]
    :return: Dict, looks like: {"0011_000001.wav": "The nine the eggs, I keep.",
                                "0011_000002.wav": "I did go, and made many prisoners."}
    """
    result_dict = {}
    with open(txt_path, "r", encoding=encoding_type) as f:
        lines = f.readlines()
        for line in lines:
            string = line.split("\t")
            print(string)
            # hard coding fix, somehow first sample from txt is read like '\ufeff0012_000001'
            if string[0] == "\ufeff0012_000001":
                result_dict["0012_000001.wav"] = string[1]

            elif string[0] == "\ufeff0013_000001":
                result_dict["0013_000001.wav"] = string[1]

            elif string[0] == "\ufeff0014_000001":
                result_dict["0014_000001.wav"] = string[1]

            elif string[0] == "\ufeff0018_000001":
                result_dict["0018_000001.wav"] = string[1]

            elif string[0] == "\ufeff0019_000001":
                result_dict["0019_000001.wav"] = string[1]

            # handle handwritten indent
            elif len(string) > 1:
                result_dict[f"{string[0]}.wav"] = string[1]
    return result_dict


def write_txt(text: str, path: str) -> None:
    with open(path, 'a') as f:
        f.write(text + "\n")


@hydra.main(config_path="configs", config_name="parallel_dataset")
def build_dataset(cfg):
    manifest_path = cfg.target_directory_path + "/parallel_manifest"
    wavs_path = cfg.target_directory_path + "/wavs"
    emotion_dict = dict(zip(cfg.emotions, cfg.emotion_ids))
    encoding_dict = dict(zip(cfg.original_speaker_ids, cfg.speaker_encodings))
    target_speaker_id_dict = dict(zip(cfg.original_speaker_ids, cfg.target_speaker_ids))
    for speaker_id in tqdm(cfg.original_speaker_ids):
        # print(cfg.original_speaker_ids)
        target_speaker_id = target_speaker_id_dict[speaker_id]
        speaker_encoding_type = encoding_dict[speaker_id]
        # print(f"{cfg.source_data_directory}/{speaker_id}/{speaker_id}.txt")
        speaker_transcripts_dict = get_speaker_transcripts(
            f"{cfg.source_data_directory}/{speaker_id}/{speaker_id}.txt", encoding_type=speaker_encoding_type
        )
        # print(speaker_transcripts_dict.keys())
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
                    # skip audio for which transcription is absent
                    if audio_id != "0014_001590":
                        continue
                    emotion_id = emotion_dict[emotion]
                    new_absolute_wav_path = f"{wavs_path}/{target_speaker_id}_{audio_id}_{emotion_id}.wav"
                    new_relative_wav_path = f"vk_etts_data/wavs/{target_speaker_id}_{audio_id}_{emotion_id}.wav"
                    shutil.copyfile(wav, new_absolute_wav_path)
                    _, wav_filename = os.path.split(wav)
                    new_txt_path = f"{wavs_path}/{target_speaker_id}_{audio_id}_{emotion_id}.txt"
                    transcription = speaker_transcripts_dict[wav_filename]
                    # write transcription, file name of txt == file name of wav for future TextGrids generation
                    write_txt(transcription, new_txt_path)
                    # write data to Manifest: "/path/to/audio.wav"|"speaker_id"|"emotion_id"|"text"
                    write_txt(
                        f"{new_relative_wav_path}|{target_speaker_id}|{emotion_id}|{transcription}", manifest_filename
                    )
    print(f"Saved wavs and manifests in {cfg.target_directory_path} folder!")


if __name__ == "__main__":
    build_dataset()
