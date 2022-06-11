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
-  if script crashes with `KeyError` for some .wav file – open relative .txt file and add tab between audio-id and transcription.

