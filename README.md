# DAS WhisperSeg

Fork of [WhisperSeg](https://github.com/nianlonggu/WhisperSeg) for use with [DAS](https://github.com/janclemenslab/das).

## Install

```shell
conda create -n wseg python=3.13 uv -y
conda activate wseg
uv pip install -e . --upgrade
```

## Usage

```shell
das-whisper train -h
das-whisper predict -h
```
