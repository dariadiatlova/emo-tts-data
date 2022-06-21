import glob
import shutil
import os
from collections import defaultdict

import hydra
import torchaudio

from typing import Dict, Tuple, List
from tqdm import tqdm

from utils import write_txt, audio_write


def get_audio_data(txt_path: str, target_emotions: List[str],
                   encoding_type: str = "us-ascii") -> Tuple[Dict, Dict, List]:
    result_data_dict = {}
    unused_data_dict = defaultdict(lambda: 0)
    with open(txt_path, "r", encoding=encoding_type) as f:
        lines = f.readlines()
        current_idx = 2
        idx_space = 9
        previous_end_time = 0.0
        overlapped_cases_found = []
        upper_bound = len(lines) - 1
        while current_idx <= upper_bound:
            line = lines[current_idx]
            times = line[0]
            emotion = line[2]
            t1, t2 = times.split(" - ")
            time_code = (float(t1[1:]), float(t2[:-1]))
            if emotion in target_emotions:
                result_data_dict[time_code] = emotion
            else:
                unused_data_dict[emotion] += 1
            current_idx += idx_space
            if float(t1[1:]) < previous_end_time:
                overlapped_cases_found.append(previous_end_time - (float(t1[1:])))
            previous_end_time = float(t2[:-1])
        return result_data_dict, unused_data_dict, overlapped_cases_found


def update_logger(emotion_logger: Dict, result_data_dict: Dict, unused_data_dict: Dict) -> Dict:
    emotions_to_write = list(result_data_dict.keys())
    unused_emotions = list(unused_data_dict.keys())
    # count emotion distribution
    for emotion in emotions_to_write:
        emotion_logger[emotion] += 1
    emotion_logger["total_wavs_count"] += len(emotions_to_write) + len(unused_emotions)
    for k, v in zip(unused_data_dict.keys(), unused_data_dict.values()):
        emotion_logger[k] = v
    return emotion_logger


@hydra.main(config_path="configs", config_name="iemocap_dataset")
def build_dataset(cfg):
    emotion_logger = defaultdict(lambda: 0)
    overlapped_logger = {}
    for session in cfg.sessions:
        time_alignment_txt_path = f"{cfg.source_data_directory}/{session}/dialog/EmoEvaluation"
        dialog_paths = glob.glob(f"{time_alignment_txt_path}/*txt")
        for dialog in dialog_paths:
            result_data_dict, unused_data_dict, overlapped_cases_found = get_audio_data(dialog, cfg.emotions)
            overlapped_logger[dialog] = overlapped_cases_found
            emotion_logger = update_logger(emotion_logger, result_data_dict, unused_data_dict)



if __name__ == "__main__":
    build_dataset()