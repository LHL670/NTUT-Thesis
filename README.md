# CVM-VIGOR: Transformer-based Cross-View Geo-Localization

This project implements CVM-VIGOR (Cross-View Matching for VIGOR), a deep learning model designed for cross-view geo-localization. The primary objective is to accurately match ground-level panoramic images (Street View Panoramas) with their corresponding satellite imagery.

The model utilizes a dual-branch EfficientNet as its backbone, enhanced by a Transformer-based fusion block for feature interaction and refinement. The final output is a heatmap that predicts the relative location of the ground-level image within the satellite view.

## Key Features

  * **Cross-View Feature Fusion**: At the core of the model is a `TransformerFusionBlock` that leverages a Cross-Attention mechanism to effectively fuse features from both ground and satellite perspectives.
  * **High-Performance Backbone**: The model employs the lightweight and efficient `EfficientNet` for feature extraction, customized with Circular Padding to handle the panoramic nature of ground-level images.
  * **Domain Generalization**: Integrates `MixStyle`, a technique that stochastically mixes feature statistics between instances during training to improve the model's generalization capabilities to new, unseen domains.
  * **Semantic Information as Guidance**: The model can optionally incorporate a semantic mask, providing it with additional structural information about the scene to aid the localization process.
  * **Comprehensive Visualization Tools**: A suite of visualization scripts is provided to generate prediction heatmaps, attention maps, and Grad-CAM outputs, facilitating in-depth model analysis and debugging.

## File Structure

```
NTUT-Thesis/
│
├── dataset/
│   └── datasets_wicaf_wDA_wmask.py       # VIGOR Dataset handler
│
├── efficientnet_pytorch/                 # EfficientNet backbone
│   ├── model.py
│   └── utils.py
│
├── loss/
│   └── losses_wDA.py                     # Loss function definitions
│
├── model/
│   ├── models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle.py         # Main model architecture
│   ├── models_wicaf_wlum_adapted2d_wDA_wmask_wmixstyle_gradcam.py # Model version with Grad-CAM support
│   └── mixstyle.py                       # MixStyle module
│
├── runs/                                 # Stores pretrained model weights
│   └── ...
│
├── train_VIGOR_wmask_wmixstyle.py        # Training script
├── eval_VIGOR_wmask_wmixstyle.py         # Evaluation script
│
├── visualization_cross_attenttion.py     # Cross-attention visualization script
├── visualization_png_loc_womask.py       # Localization result visualization script
├── visualization_png_semi_positive.py    # Semi-positive sample visualization script
│
└── environment.yaml                      # Conda environment configuration
```

## Setup and Installation

### 1\. Create Conda Environment

All dependencies for this project are listed in the `environment.yaml` file. Use Conda to create and activate the environment:

```bash
# Create the conda environment from the YAML file
conda env create -f environment.yaml

# Activate the environment
conda activate crossview
```

### 2\. Download the VIGOR Dataset

To get the VIGOR dataset, you must apply for access through the official GitHub repository. Follow the instructions on the site to submit a request:
[**https://github.com/Jeff-Zilence/VIGOR**](https://github.com/Jeff-Zilence/VIGOR)

Once you have access, download the data and extract it.

### 3\. Prepare the Dataset (Generate Semantic Masks)

This project uses semantic masks as an additional input. To generate these masks, please use our other repository, which leverages the Segment Anything Model (SAM):
[**https://github.com/LHL670/NTUT-Thesis-SAM-Geospatial**](https://github.com/LHL670/NTUT-Thesis-SAM-Geospatial)

Follow the instructions in that repository to process the VIGOR satellite images and generate the corresponding masks. After generation, your dataset directory should be structured as follows:

```
<dataset_root>/
├── Chicago
│   ├── panorama
│   ├── point_prompt_mask
│   └── satellite
├── NewYork
│   ├── ...
├── SanFrancisco
│   ├── ...
├── Seattle
│   ├── ...
└── splits__corrected
    ├── Chicago
    ├── NewYork
    ├── SanFrancisco
    └── Seattle
```

> **Note**: `splits__corrected` contains the revised ground truth from the [SliceMatch repository](https://github.com/tudelft-iv/SliceMatch).

Finally, update the `dataset_root` variable in the scripts to point to your dataset's location.

### 4\. Download Model Weights

Pre-trained model weights are available for download from the following link:

[**Model Weights - Google Drive Link**](https://drive.google.com/drive/folders/1IPFG9aiSMuUScdLueBfIRQiqF96qoXkY?usp=sharing)

After downloading, extract the `runs` folder and place it in the root directory of this project.

## Usage

### Training

To train the model from scratch, execute the `train_VIGOR_wmask_wmixstyle.py` script. It supports several command-line arguments for configuration:

```bash
python train_VIGOR_wmask_wmixstyle.py \
    --area 'samearea' \
    --batch_size 10 \
    --learning_rate 8e-5 \
    --epochs 120 \
    --d_model 320 \
    --n_layer 2 \
    --use_mask True \
    --use_mixstyle True \
    --mixstyle_mix 'random' \
    --dataset_root <path_to_your_dataset_root>
```

  * `--area`: Choose between 'samearea' or 'crossarea' training modes.
  * `--use_mixstyle`: A boolean flag to enable or disable MixStyle.
  * `--mixstyle_mix`: The MixStyle strategy, either 'random' or 'crossdomain'.
  * `--use_mask`: A boolean flag to enable or disable the use of semantic masks during training.

The trained model weights and configuration files will be saved under the `runs/VIGOR/` directory.

### Evaluation

To evaluate a trained model, run the `eval_VIGOR_wmask_wmixstyle.py` script and specify the path to your model checkpoint:

```bash
python eval_VIGOR_wmask_wmixstyle.py \
    --model_path <path_to_your_model.pt> \
    --batch_size 2 \
    --dataset_root <path_to_your_dataset_root>
```

Evaluation results, including mean/median errors, CED curves, and a detailed CSV report, will be saved in a `results/` folder within the model's directory.

### Visualization

This project includes several scripts to visualize the model's behavior and predictions:

1.  **Localization Heatmap Visualization**
    Generates a visual summary for a single data sample, including the original images and the predicted heatmap.

    ```bash
    python visualization_png_loc_womask.py \
        --model_path <path_to_your_model.pt> \
        --idx 1 \
        --show_mask
    ```

      * `--idx`: The index of the validation set image to visualize.
      * `--show_mask`: If specified, the output will include a visualization of the input semantic mask.

2.  **Cross-Attention Visualization**
    This script generates Grad-CAM outputs and Transformer attention maps to analyze the model's decision-making process.

    ```bash
    python visualization_cross_attenttion.py \
        --resume <path_to_your_model.pt> \
        --img_idx 1
    ```

3.  **Semi-Positive Sample Visualization**
    This script is designed to analyze the model's predictions for one ground image against its four corresponding satellite views (one positive, three semi-positive).

    ```bash
    python visualization_png_semi_positive.py \
        --model_path <path_to_your_model.pt> \
        --idx 1
    ```

All visualization outputs will be saved in a `visualizations_*` folder inside the corresponding model checkpoint directory.