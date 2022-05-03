# Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Usage
## Dependency
The code is tested with fairseq commit bba000d.
```
torch==1.9.0+cu111
torchaudio==0.9.0
tqdm==4.62.3
hydra-core==1.0.7
omegaconf==2.0.6
einops==0.3.0
fire==0.4.0
fairseq==1.0.0a0+bba000d
```

## Installation
```
git clone git@github.com:asappresearch/wav2seq.git
cd wav2seq
pip install -e .
```

## Creatining Psuedo Subword Tokens
1. Create wav2vec style manifest files
Please set `LIBRISPEECH_PATH` to your librispeech folder which contains three subfolders `train-clean-100`, `train-clean-360`, `train-other-500`.
```sh
mkdir -p manifest/librispeech/train-960
python -m examples.wav2vec.wav2vec_manifest LIBRISPEECH_PATH  --dest manifest/librispeech/train-960 --ext flac --valid-percent 0.01 --path-must-contain train
```

2. Train k-means model and get cluster indices
Please make sure that you have download pre-trained hubert-base checkpoint at `HUBERT_PATH`.
Notably, this step requires a GPU for feature extraction and 64GB main memory for k-means training.
Extracting HuBERT features takes about 15 minutes, training k-means may take about an hour, dumping the cluster ids of the whole Librispeech 960h data takes more than two hours.
```sh
HUBERT_PATH="save/pretrained/hubert_base_ls960.pt"
mkdir -p save/pretrained
if ! [ -f $HUBERT_PATH ]; then
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -O $HUBERT_PATH
fi
bash scripts/pl/extract-hubert-features.sh $HUBERT_PATH 9 2 2 500
```
where 9, 2, 2, 500 means that we use the 9-th layer of HuBERT, kernel size 2 and stride size 2 for average pooling, and 500 custers in k-means.

3. Training BPE model and create pseudo subword tokens
```sh
bash scripts/pl/create-hubert-pseudo-language.sh labels/hubert_base-l9-k2s2-fp16-ls0.1/c500 30000
```



## Pre-training Wav2Seq

## Fine-tuning Wav2Seq

# Pre-trained Models
Our pretrained models will be released.