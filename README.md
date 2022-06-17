# ETTS data preprocessing :blossom:

This repository consists of supprorting scripts for building datasets for ETTS in English 

### Repository structure: 
- `src` folder consists of scripts and `configs` for processing different datasets;
- all data is processed to create uniform data structure in `target_directory_path` directory (path is set in relative config file).

### Expected files structure:

    - target_directory_path:
       - wavs:
          - {speaker_id}_{audio_id}_{emotion_id}.wav
          - ...
          - {speaker_id}_{audio_id}_{emotion_id}.txt
       - {dataset_name}_manifest_train.txt:
          - wav_path|speaker_id|emotion_id|transcription
       - {dataset_name}_manifest_evaluation.txt
       - {dataset_name}_manifest_test.txt
       
       
### Datasets:

#### 1. Parallel ESD
- [Downloading link](https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view)
- [Arxiv](https://arxiv.org/pdf/2010.14794.pdf)

To process English-speakers data:
-  fill `source_data_directory` in [config.yaml](src/configs/parallel_dataset.yaml) with the root path to the dataset;
-  run [parallel_dataset](src/parallel_dataset.py);
-  if script crashes with `KeyError` for some .wav file – open relative .txt file and add tab between audio-id and transcription.


#### 2. EmovDB 
- [Downloading wavs](https://openslr.org/115/), [Download transcriptions](http://www.festvox.org/cmu_arctic/cmuarctic.data)
- [Arxiv](https://arxiv.org/pdf/1806.09514.pdf)

We use only 2 emotions: `Neutral`, `Angry` taken from 3 speakers: Bea, Jenie, Sam. To process data:
- unarchive relevant `tar.gz` folders and point the directory in `source_data_directory` in [config.yaml](src/configs/emodb_dataset.yaml);
- run [emodb_dataset](src/emodb_dataset.py);
- place `cmuarctic.data` in the same `source_data_directory`.


### Profiler:
To get stats (emotion distribution per hour and total dataset size): fill [profiler.yaml](src/configs/profiler.yaml) and run [profiler.py](src/configs/profiler.py).

Output:

        Emotion 'Neutral': 2.98 hours.
        Emotion 'Sad': 2.07 hours.
        Emotion 'Angry': 3.313 hours.
        Emotion 'Happy': 1.845 hours.
        Emotion 'Surprised': 1.869 hours.
        Total dataset size: 12.077 hours.


