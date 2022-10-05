import os
import numpy as np
from tqdm import tqdm
from speechbrain.pretrained.interfaces import foreign_class


def esd_compute_scores(wav_dir_path: str):
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                               pymodule_file="custom_interface.py",
                               classname="CustomEncoderWav2vec2Classifier")

    wavs = os.listdir(wav_dir_path)
    acc_dict = {"0": [], "1": [], "2": [], "3": []}
    for wav in tqdm(wavs):
        if ".wav" not in wav:
            continue
        s, i, e = wav[:-4].split("_")
        if e == "4":
            continue
        full_path = os.path.join(wav_dir_path, wav)
        _, _, index, _ = classifier.classify_file(full_path)
        if e == index:
            acc_dict[e].append(1)
        else:
            acc_dict[e].append(0)

    neu_score = np.mean(acc_dict['0'])
    ang_score = np.mean(acc_dict['1'])
    hap_score = np.mean(acc_dict['2'])
    sad_score = np.mean(acc_dict['3'])

    print(f"Accuracy for neutral audios: {neu_score}")
    print(f"Accuracy for angry audios: {ang_score}")
    print(f"Accuracy for happy audios: {hap_score}")
    print(f"Accuracy for sad audios: {sad_score}")

    print(f"Overall acc: {np.mean([neu_score, ang_score, hap_score, sad_score])}")


if __name__ == "__main__":
    esd_compute_scores("/root/storage/dasha/data/emo-data/esd/wavs")
