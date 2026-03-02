# DAS WhisperSeg

Fork of [WhisperSeg](https://github.com/nianlonggu/WhisperSeg) for use with [DAS](https://github.com/janclemenslab/das).

## Install

```shell
conda create -n das-wseg python=3.13 uv -y
conda activate das-wseg
uv pip install git+https://github.com/janclemenslab/das-whisper --upgrade
```


## Use
Train/evaluate/predict using the CLI interface:
```shell
das-whisper train -h
das-whisper predict -h
das-whisper evaluate -h
```

To convert your own annotations for use with whisperseg, see the original [docs](https://github.com/nianlonggu/WhisperSeg/blob/master/docs/DatasetProcessing.md).


Complete usage example:
### Train
```shell
das-whisper train nccratliri/whisperseg-base-animal-vad models/bf nccratliri/bengalese-finch-subset-with-csv-label/train --val-ratio 0.2 --validate-every 100 --save-every 100
```
This will download the dataset [nccratliri/bengalese-finch-subset-with-csv-label](https://huggingface.co/datasets/nccratliri/bengalese-finch-subset-with-csv-label) from huggingface, train a whisperseg model using the training set and using the pre-trained model [nccratliri/whisperseg-base-animal-vad](https://huggingface.co/nccratliri/whisperseg-base-animal-vad) as a starting point. The best trained model will be saved to `models/bf/final_checkpoint`.

### Evaluate
```shell
das-whisper evaluate nccratliri/bengalese-finch-subset-with-csv-label/test models/bf/final_checkpoint
```
This will evaluate the best model on the test data in the dataset folder.

### predict
```shell
das-whisper predict nccratliri/bengalese-finch-subset-with-csv-label/test models/bf/final_checkpoint nccratliri/bengalese-finch-subset-with-csv-label/test_predictions.csv
```
This will use to trained model to create annotations files for the test data in `results`.

