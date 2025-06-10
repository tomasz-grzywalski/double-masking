# Salient Sound Detection and Extraction Using Double Masking

## What is this repository for?

This repository contains scripts for training a deep convolutional neural network to extract salient sounds using a novel double masking approach. A publication describing this method is currently in preparation and will be made available soon.

This is an updated and unpublished version of the research presented in: https://doi.org/10.23919/SPA61993.2024.10715626.

## Setup

1. Download the dataset of salient sounds from [Zenodo](https://zenodo.org/records/15631023).
2. Install the dependencies (these were tested; newer versions may work too):
   * `librosa>=0.8.0,<=0.10.0.post2`
   * `numpy>=1.19.5,<=1.22.4`
   * `tensorflow-gpu>=2.5.0,<=2.12.0`
   * `matplotlib==3.4.2`
   * `scipy>=1.7.1,<=1.10.1`
   * `pandas==1.3.1`

## Repository Structure and Main Functions

### Dataset Creation

The dataset used in this project can be downloaded from: https://zenodo.org/records/15631023

If you'd like to create your own dataset or examine how the original was prepared, refer to the `dataset_creator` directory. It contains scripts used to process unlabeled field recordings from [freesound.org](https://freesound.org) into datasets of:
- Backgrounds (salient sound-free)
- Foregrounds (isolated salient sounds)

The process uses a model trained on AudioSet to check consistency of audio embeddings in a sliding window fashion. The `model` subdirectory contains the architecture (pretrained weights not included due to size).

### Training

1. Edit `train.py`:
   - Set `DATASET_ROOT` to the root directory of your dataset, which should include `Background/` and `Foreground/` folders.
   - Set `OUTPUT_ROOT` to the directory where model weights will be saved.

2. Run `train.py`. Logs are printed to stdout — consider redirecting them to a file.

**Optional:**  
Create a folder named `DUMPS` in the root directory. Then, in `utils/dataset.py`, set `VISUALIZE_DATASET = True` to generate visualizations of all generated mixtures. Stop the process manually when enough samples are visualized.

### Testing

The test script outputs wave files for:
- Clean salient sound
- Mixture with background
- Extracted salient sound

It can also generate visualizations.

To use:

1. Edit `test.py` and set `DATASET_ROOT` to your dataset's root directory.

2. Run `test.py` with the following options:
   * `--snr`  
     Signal-to-noise ratio of the test mixtures. Use `>=100` to test with clean foregrounds only.
   * `--model_dir`  
     Directory containing trained model weights.
   * `--out_dir`  
     Output directory for wave files and visualizations.
   * `--visualize`  
     Add this flag to enable visualization generation.

## Contact

If you use this repository, please contact the author or cite the relevant work:

**Tomasz Grzywalski**  
📧 tomasz.grzywalski@gmail.com  

References:
- [Conference Paper](https://doi.org/10.23919/SPA61993.2024.10715626)
- [Dataset on Zenodo](https://zenodo.org/records/15631023)
