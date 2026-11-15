# Risk–Coverage–Accuracy for Clinica AI Screening

This repository provides a PyTorch pipeline for training retinal image classifiers and evaluating uncertainty quantification (UQ) methods under a deployment-oriented **risk–coverage–accuracy** framework. The code supports standard image classification, deep ensembles, test-time augmentation, MC dropout, entropy-based uncertainty, maximum softmax probability, and conformal prediction.

The current examples focus on binary retinal disease classification tasks such as AMD, diabetic retinopathy, and glaucoma referral using fundus images.

---

## Main Features

- Train binary retinal image classifiers using `timm` backbones.
- Support for ViT/DINOv2, ConvNeXt, and RETFound-Green-style models.
- Weighted cross-entropy and focal loss for class imbalance.
- Deep ensemble training for uncertainty estimation.
- Inference with multiple UQ methods:
  - No UQ / standard prediction
  - Maximum softmax probability (MSP)
  - Predictive entropy
  - Tsallis predictive entropy
  - MC dropout
  - Test-time augmentation (TTA)
  - Deep ensembles
- Conformal prediction as an additional uncertainty/safety layer.
- Evaluation scripts for calibration, AUROC/AUPRC, risk–coverage curves, PPV/NPV, disagreement analysis, and uncertainty visualizations.

---

## Repository Structure

```text
Risk-Coverage-Accuracy/
├── main_train.py                         # Model training
├── main_inference_step1_getPredictions.py # Prediction + uncertainty score generation
├── main_inference_step2_conformal.py      # Conformal prediction post-processing
├── utils.py                              # Argument parsing and preprocessing utilities
├── utils_data.py                         # Dataset and transform utilities
├── utils_models.py                       # Model definitions, training loop, focal loss
├── utils_uncertainty.py                  # UQ methods
├── utils_metrics.py                      # Metric utilities
├── requirements.txt                      # Python dependencies
├── TrainTestCSVfiles/                    # Example train/test CSV split files
├── scripts4evaluations/                 # Evaluation and plotting scripts
└── data/                                 # Example images
```

---

## Installation

Create a new environment and install the dependencies:

```bash
conda create -n rca-retina python=3.10 -y
conda activate rca-retina
pip install -r requirements.txt
```

Core dependencies include:

```text
torch
torchvision
timm
transformers
pandas
numpy
scikit-learn
scipy
matplotlib
Pillow
tqdm
requests
huggingface-hub
safetensors
```

For GPU training, install the PyTorch version compatible with your CUDA version if the pinned version in `requirements.txt` does not match your system.

---

## Input CSV Format

Training and inference expect a CSV file with at least the following columns:

| Column | Required For | Description |
|---|---|---|
| `image_path` | training and inference | Full or relative path to the retinal image. |
| `label` | training | Binary class label, usually `0` or `1`. |
| `tag_4calibrationSet` | conformal prediction | Use `1` for calibration samples and `0` for test samples. Required by `main_inference_step2_conformal.py`. |

Example:

```csv
image_path,label,tag_4calibrationSet
/path/to/image_001.jpg,0,1
/path/to/image_002.jpg,1,0
/path/to/image_003.jpg,0,0
```

Example split files are provided in `TrainTestCSVfiles/`.

---

## Training

### Basic Training

```bash
python main_train.py \
  --experiment_name AMD_ViT_single \
  --csv_path TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_AMD_train.csv \
  --model_name ViT \
  --train_strategy none \
  --batch_size 250 \
  --num_epochs 150
```

Trained weights are saved under:

```text
trainedModels/<experiment_name>/
```

---

### Deep Ensemble Training

```bash
python main_train.py \
  --experiment_name AMD_ViT_ensemble_1 \
  --csv_path TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_AMD_train.csv \
  --model_name ViT \
  --train_strategy deepensemble \
  --num_ensemble_models 5 \
  --batch_size 250 \
  --num_epochs 150
```

This creates multiple independently trained model checkpoints in the experiment folder.

---

### Training with Focal Loss

```bash
python main_train.py \
  --experiment_name DR_ViT_ensemble_focal \
  --csv_path TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_DR_train.csv \
  --model_name ViT \
  --train_strategy deepensemble \
  --num_ensemble_models 5 \
  --loss_type focal \
  --focal_schedule flsd53 \
  --batch_size 250 \
  --num_epochs 150
```

---

## Training Arguments

| Argument | Type | Default | Possible Values / Description |
|---|---:|---:|---|
| `--experiment_name` | `str` | `experiment_default` | Name used to save trained models and results. |
| `--csv_path` | `str` | `./data.csv` | Path to the training CSV file. |
| `--model_name` | `str` | `ViT` | `ViT`, `convnext`, `RETFoundGreen`. |
| `--train_strategy` | `str` | `none` | `none`, `deepensemble`, `evidential`. Current common workflow uses `none` or `deepensemble`. |
| `--num_ensemble_models` | `int` | `3` | Number of models for deep ensemble training. |
| `--num_classes` | `int` | `2` | Number of output classes. Current scripts are mainly designed for binary classification. |
| `--image_size` | `int` | `392` | Input image size used for resizing/cropping. |
| `--batch_size` | `int` | `3` | Training batch size. |
| `--num_epochs` | `int` | `20` | Maximum number of training epochs. |
| `--learning_rate` | `float` | `0.0005` | Adam optimizer learning rate. |
| `--patience_epochs_LRplateau` | `int` | `10` | Patience for `ReduceLROnPlateau`. |
| `--patience_epochs_early_stop` | `int` | `15` | Early stopping patience. |
| `--loss_type` | `str` | `ce` | `ce` or `focal`. |
| `--focal_gamma` | `float` | `3.0` | Gamma for focal loss when using the fixed schedule. |
| `--focal_schedule` | `str` | `fixed` | `fixed` or `flsd53`. |

---

## Inference and Uncertainty Quantification

After training, run inference using one or more UQ methods:

```bash
python main_inference_step1_getPredictions.py \
  --experiment_path ./trainedModels/DR_ViT_ensemble_1 \
  --csv_path ./TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_DR_test.csv \
  --UQ_method None MSP PredictiveEntropy PredictiveEntropy_tsallis McDropout TTA DeepEnsemble
```

Outputs are saved under:

```text
Outputs/<experiment_name>/predictions_<UQ_method>.csv
```

The output CSV files include columns such as:

| Column | Description |
|---|---|
| `prediction` | Predicted class label. |
| `prob_class_1` | Predicted probability for class 1. |
| `uncertainty` | Main uncertainty score, depending on the selected UQ method. |
| `uncertainty_class_1` | Class-1-specific uncertainty score for applicable methods. |
| `uncertainty_epistemic` | Epistemic uncertainty for deep ensembles. |
| `uncertainty_aleatoric` | Aleatoric uncertainty for deep ensembles. |
| `note` | Description of the inference/UQ setting. |

---

## Inference Arguments

| Argument | Type | Default | Possible Values / Description |
|---|---:|---:|---|
| `--csv_path` | `str` | `./data.csv` | Path to the inference/test CSV file. |
| `--experiment_path` | `str` | `./trainedModels/experiment_default` | Folder containing trained `.pth` checkpoint files. |
| `--UQ_method` | `str`, one or more | `None` | `None`, `MSP`, `PredictiveEntropy`, `PredictiveEntropy_tsallis`, `McDropout`, `TTA`, `DeepEnsemble`, `MSP_tuned`, `PredictiveEntropy_tuned`, `PredictiveEntropy_tsallis_tuned`, `Conformal`. |
| `--MC_DROPOUT_SAMPLES` | `int` | `50` | Number of stochastic forward passes for MC dropout. |
| `--TTA_SAMPLES` | `int` | `20` | Number of augmented views for TTA. |
| `--num_classes` | `int` | `2` | Number of classes. |
| `--image_size` | `int` | `392` | Input image size. |

---

## Conformal Prediction

Conformal prediction is applied as a second step to the prediction CSV files generated above.

Currently, edit the following line inside `main_inference_step2_conformal.py`:

```python
path_savedPredictions = "./Outputs/FocalLoss/GL_ViT_ensemble_1"
```

Then run:

```bash
python main_inference_step2_conformal.py
```

This creates new files with the suffix:

```text
_conformal.csv
```

The conformal output includes:

| Column | Description |
|---|---|
| `conf_pred_set` | Conformal prediction set, e.g., `[0]`, `[1]`, or `[0, 1]`. |
| `set_sizes` | Size of the prediction set. |
| `true_label_in_set` | Whether the reference label is included in the conformal set. |
| `isUncertain` | `0` for confident singleton sets, `1` for uncertain two-class sets. |
| `Conformal Threshold` | Estimated conformal threshold. |
| `note_conformal` | Conformal configuration note. |

Important: the current conformal script expects the prediction CSV to contain `tag_4calibrationSet`, where `1` indicates calibration samples and `0` indicates test samples.

---

## Evaluation and Figure Generation

Evaluation scripts are available in `scripts4evaluations/`:

| Script | Purpose |
|---|---|
| `IMPO_AUROCAUPRCCurveGenerator.py` | AUROC/AUPRC curve generation. |
| `IMPO_CalibrationCurveGenerator.py` | Calibration curve generation. |
| `IMPO_Risk-coverageAnalysis.py` | Risk–coverage analysis. |
| `IMPO_UncertaintyViolin.py` | Uncertainty distribution visualization. |
| `IMPO_conformalAnalysisv4.py` | Conformal prediction analysis. |
| `IMPO_uncertainty_conformal_show.py` | Visualization of uncertainty and conformal outputs. |
| `IMPO_disagreement_v2_drishti.py` | Disagreement analysis. |
| `ppv_npvAnalysisV4.py` | PPV/NPV analysis. |

Some evaluation scripts use hard-coded paths. Update the input/output paths inside each script before running.

---

## Suggested End-to-End Workflow

```bash
# 1. Train an ensemble model
python main_train.py \
  --experiment_name GL_ViT_ensemble_1 \
  --csv_path TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_Gl_train.csv \
  --model_name ViT \
  --train_strategy deepensemble \
  --num_ensemble_models 5 \
  --batch_size 250 \
  --num_epochs 150

# 2. Generate predictions and uncertainty scores
python main_inference_step1_getPredictions.py \
  --experiment_path ./trainedModels/GL_ViT_ensemble_1 \
  --csv_path ./TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_Gl_test.csv \
  --UQ_method None MSP PredictiveEntropy PredictiveEntropy_tsallis McDropout TTA DeepEnsemble

# 3. Apply conformal prediction
python main_inference_step2_conformal.py

# 4. Run evaluation/plotting scripts from scripts4evaluations/
```

---

## Notes and Current Limitations

- The current code is mainly configured for binary classification.
- Some evaluation and conformal scripts contain hard-coded paths that should be edited before use.
- Deep ensemble inference expects multiple `.pth` files inside the selected experiment folder.
- RETFound-Green weights may be downloaded automatically when using `--model_name RETFoundGreen`.
- The training code freezes the backbone and trains the classification head by default.
- CSV paths can be absolute or relative, but all image paths must be valid from the directory where the command is executed.

---

## Citation

If you use this repository, please cite the associated paper or manuscript:

```bibtex
@article{your_citation_here,
  title   = {Risk-Coverage-Accuracy Evaluation for Uncertainty Quantification in Retinal AI Screening},
  author  = {Your Authors},
  journal = {TBD},
  year    = {2026}
}
```

---

## License

This repository includes a `LICENSE` file. Please check it before using or redistributing the code.
