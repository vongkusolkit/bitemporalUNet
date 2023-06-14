# Near Real-Time Flood Mapping with Weakly-Supervised Machine Learning

This repository contains the implementation of the framework for detecting flooded pixels
for flood mapping without human-labeled data. The framework consists of two main parts: 
1) Generating a weakly-labeled flood map to train the model and 
2) Implementing the bitemporal UNet architecture

## Prerequisites
Install the required dependencies with the following command:
```
pip install -r requirements.txt
```

## Part 1: Weakly-labeled flood map
1. Get NDWI difference image from pre- and post-flood images
2. Binary classification of the NDWI difference image using either histogram thresholding or k-means clustering (`thresholding_types_florence.ipynb`)
3. Remove noise through edge detection (edge_detection_harvey.ipynb)

## Part 2: Bitemporal UNet

### Training the Model
To train the model (see `train.py`), you need:
1. Directory containing tiles of pre-flood images
2. Directory containing tiles of post-flood images
3. Directory containing tiles of weakly-labeled flood map (generated from Part 1) for "ground truth"
4. CSV files of the training data filenames
5. CSV files of the validation data filenames

Change the parameters specified in `train.py` using argument parser:
* `--version`: version name
* `--t`: training batch size
* `--v`: validation batch size
* `--e`: number of epochs
* `--lr`: learning rate
* `--wd`: weight decay
* `--csv_train`: csv file containing training data filenames
* `--csv_valid`: csv file containing validation data filenames
* `--data_root_dir_pre`: directory of pre-flood images
* `--data_root_dir_post`: directory of post-flood images
* `--data_root_dir_gt`: directory of weakly-labeled flood map

### Testing the Model
To test the model and get the predictions, run `test.py` with your defined parameters.
Model predictions are binarized with either histogram thresholding or k-means clustering (`thresholding_types_florence.ipynb`) for evaluation
