# Baseline
## Prerequisites

Before starting, ensure you have Python installed on your system. Then install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create an Ananconda environment
```
conda create -n sl_env python=3.9
conda activate sl_env
```
2. Install the required packages
```
pip install -r requirements.txt
```
3. Create an .env file in the root directory.
4. Add your Weights & Biases API key:
```
WANDB_API_KEY=YOUR_WANDB_API_KEY
```

## Data Preparation

### Frame Extraction

Edit `scripts/extract_frames.sh`:
   - Replace `INPUT_FOLDER` with your video directory
   - Replace `OUTPUT_BASE_FOLDER` with your desired output frame directory

```bash
bash extract_frames.sh
```

### Dataset Splitting

For each frame folder, run the dataset splitting script:

```bash
bash split_datasets.sh
```

Before running, modify the script parameters:
- `SOURCE_DIR`: Directory containing the extracted frames
- `OUTPUT_DIR`: Directory where the split datasets will be saved

## Training

1. Edit the following fields in the model configuration & shell script:
   - `TRAIN_DATA_PATH`: Path to training data folder
   - `VAL_DATA_PATH`: Path to validation data folder

2. Start training by running the scripts in /scripts folder: `videomae_v2.sh` for VideoMAE, `x3d.sh` for X3D and `i3d.sh` for I3D.

