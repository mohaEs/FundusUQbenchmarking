# Risk-Coverage-Accuracy



| Argument | Type | Default | Possible Values / Description |
|---|---|---|---|
| `--experiment_name` | `str` | `experiment_default` | Name of experiment for saving models/results |
| `--csv_path` | `str` | `./data.csv` | Path to CSV file containing image paths and labels |
| `--model_name` | `str` | `ViT` | `ViT`, `convnext`, `RETFoundGreen` |
| `--train_strategy` | `str` | `none` | `none`, `deepensemble` |
| `--num_ensemble_models` | `int` | `3` | Number of models for deep ensemble training |
| `--num_classes` | `int` | `2` | Number of output classes |
| `--image_size` | `int` | `392` | Input image size (height and width) |
| `--batch_size` | `int` | `3` | Training batch size |
| `--num_epochs` | `int` | `20` | Number of training epochs |
| `--learning_rate` | `float` | `0.0005` | Optimizer learning rate |
| `--patience_epochs_LRplateau` | `int` | `10` | Patience for ReduceLROnPlateau scheduler |
| `--patience_epochs_early_stop` | `int` | `15` | Early stopping patience |
| `--loss_type` | `str` | `ce` | `ce` (CrossEntropy), `focal` |
| `--focal_gamma` | `float` | `3.0` | Gamma value for focal loss |
| `--focal_schedule` | `str` | `fixed` | `fixed`, `flsd53` |



====================TRAINING===== Without Focal Loss ============================
 
python main_train.py --batch_size 250 \
 --num_epochs 150 \
 --experiment_name AMD_ViT_ensemble_1 \
 --csv_path "TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_AMD_train.csv" \
 --model_name ViT \
--train_strategy deepensemble \
 --num_ensemble_models 5 
 
 
====================TRAINING===== With Focal Loss ============================
  
  
python main_train.py --batch_size 250 \
 --num_epochs 150 \
 --experiment_name DR_ViT_ensemble_1 \
 --csv_path "TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_AMD_train.csv" \
 --model_name ViT \
 --train_strategy deepensemble \
 --num_ensemble_models 5 \
   --loss_type focal \
  --focal_schedule flsd53



====================Inference=================================
For step 1, use the following commands.
For step2, which is conformal prediction, use the main_inference_step2_conformal.py and run it for each output directory from step 1.

step1:   
python main_inference_step1_getPredictions.py --experiment_path "./trainedModels/DR_ViT_ensemble_1" \
  --csv_path "./TrainTestCSVfiles/Fundus_filePaths_and_binaryLabels_DR_test.csv"\
  --UQ_method None MSP PredictiveEntropy PredictiveEntropy_tsallis McDropout TTA DeepEnsemble   
  
Step 2: edit the following lines in main_inference_step2_conformal.py
    path_savedPredictions = "./Outputs/CELoss/DR_ViT_ensemble_1"  # Directory containing CSV files of predictions
    process_directory(path_savedPredictions, alpha=0.1, out_suffix="_conformal.csv")
 

====================Figures and Metrics=================================

all scripts are available in scripts4evaluations folder




