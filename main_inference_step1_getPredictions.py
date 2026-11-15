import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import timm
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from utils import CustomPreprocessing
import glob
import warnings

from utils_uncertainty import find_optimal_parameters


# ====== 🔧 CONFIGURATION (Set These Paths Manually) ======
from utils import get_args_inference 
args = get_args_inference()

CSV_FILE = args.csv_path
experiment_path = args.experiment_path
uncertainty_methods= args.UQ_method
NUM_CLASSES = args.num_classes
image_size = args.image_size

print("=== uncertainty methods: ", uncertainty_methods)


MC_DROPOUT_SAMPLES = args.MC_DROPOUT_SAMPLES # Number of forward passes for MC Dropout
TTA_SAMPLES = args.TTA_SAMPLES # Number of augmented samples for TTA


trained_models_path = glob.glob(os.path.join(experiment_path, "**", "*.pth"), recursive=True)
print(len(trained_models_path))
list_models_path = []
list_models_names = []

for file_path in trained_models_path:
    list_models_path.append(file_path)
    filename = os.path.basename(file_path)  # e.g., "ViT_model.pth"
    if "_model" in filename and filename.endswith(".pth"):
        model_name = filename.split("_model")[0] # e.g., "ViT"
        #print(f"Full path: {file_path}")
        #print(f"Filename: {filename}")
        print(f"Model name: {model_name}")
        print("-" * 40)
        list_models_names.append(model_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils_models import get_model
list_models = []
for seed in range(len(list_models_path)):
    model_path = list_models_path[seed]
    model_name = list_models_names[seed]
    ### Model Initialization #######
    model = get_model(model_name, image_size, NUM_CLASSES)
    model = model.to(device)

    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)    
    list_models.append(model)

# ==========================================================

### get dataset, transforms ######
from utils_data import RetinalDataset, prepare_data, get_transforms, InferenceDataset
train_transforms, val_test_transforms, inference_transform = get_transforms(image_size)


from utils_uncertainty import get_tta_transforms
from utils_uncertainty import TTAInferenceDataset
from utils_uncertainty import mc_dropout_inference
from utils_uncertainty import tta_inference
from utils_uncertainty import ensemble_inference
from utils_uncertainty import ensemble_entropy_terms
from utils_uncertainty import predictive_entropy_inference
from utils_uncertainty import predictive_tsallis_entropy_inference
from utils_uncertainty import msp_inference

def run_inference_with_uncertainty(csv_file, model_name, list_models_path, output_csv, uncertainty_method='None'):
    """Enhanced inference function with uncertainty quantification"""
    
    df_test = pd.read_csv(csv_file)       

    if len(list_models_path) != 1:
        warnings.warn(
    f"\n ******NOTICE********* Experiment path contains more than one trained model: {experiment_path}. "
    "The first model will be used for all UQ methods except deep ensemble.",
    UserWarning
        )

    model_path = list_models_path[0]

    ### Model Initialization #######
    model = get_model(model_name, image_size, NUM_CLASSES)
    model = model.to(device)

    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
       

    predictions = []
    probabilities_list = []
    uncertainties_list = []
    uncertainties_list_class=[]
    uncertainties_epistemic_list = []
    uncertainties_aleatoric_list = []
    #uncertainties_Totalpredictive_list = []

    print("--- " * 10)
    print(f"============= > Running inference with uncertainty quantification method: {uncertainty_method} ")

    if uncertainty_method in ['TTA']:

        # Use TTA dataset
        tta_transforms = get_tta_transforms(image_size, num_tta_samples=TTA_SAMPLES)
        dataset = TTAInferenceDataset(df_test, tta_transforms)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)  # Smaller batch size for TTA
        
        model.eval()
        for augmented_images, paths in tqdm(dataloader, desc="Processing batches"):
            augmented_images = augmented_images.to(device)
            
            mean_pred, uncertainty = tta_inference(model, augmented_images)
                          
            # Get final predictions
            preds = np.argmax(mean_pred, axis=1)
            positive_probs = mean_pred[:, 1]
            uncertainty_scores = uncertainty[:, 1] if uncertainty.ndim > 1 else uncertainty
            
            predictions.extend(preds)
            probabilities_list.extend(positive_probs)
            uncertainties_list_class.extend(uncertainty_scores)
        note = f"TTA inference with {TTA_SAMPLES} augmentations"

    else:

        #################################################
        ############## # inference with test set ################
        #################################################
        
        dataset = InferenceDataset(df_test, transform=inference_transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        for images, paths in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)

            if uncertainty_method == 'None':
                model.eval()
                with torch.no_grad():
                    outputs = model(images)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    mean_pred = probabilities.cpu().numpy()
                    uncertainty = np.zeros_like(mean_pred)  # No uncertainty               
                    note = "No uncertainty quantification"
          
            elif uncertainty_method == 'MSP':
                model.eval()
                probs, uncertainty, note = msp_inference(model, images)
                mean_pred = probs

            elif uncertainty_method == 'PredictiveEntropy':
                model.eval()
                probs, H_total, H_class, note = predictive_entropy_inference(model, images)                
                mean_pred = probs
                uncertainty = H_class

            elif uncertainty_method == 'PredictiveEntropy_tsallis':
                model.eval()
                probs, H_total, H_class, note = predictive_tsallis_entropy_inference(model, images, q=0.7)                
                mean_pred = probs
                uncertainty = H_class

            elif uncertainty_method == 'McDropout':
                mean_pred, uncertainty, note = mc_dropout_inference(model, images, MC_DROPOUT_SAMPLES)                

            elif uncertainty_method == 'DeepEnsemble':
                mean_pred, uncertainty, note = ensemble_inference(list_models, images,)  
                # uncertainty class 1 will be standard deviation of all predictions
                H_tot, H_alea, MI = ensemble_entropy_terms(list_models, images,) 

            ###############################################
            #############################################
            # Get final predictions
            preds = np.argmax(mean_pred, axis=1)
            positive_probs = mean_pred[:, 1]# Probability of positive class (class 1)
            #uncertainty for positive class (class 1)
            uncertainty_scores = uncertainty[:, 1] if uncertainty.ndim > 1 else uncertainty 
            
            predictions.extend(preds)
            probabilities_list.extend(positive_probs)
            uncertainties_list_class.extend(uncertainty_scores)
            
            ###############################################
            ###############################################
            if uncertainty_method == 'DeepEnsemble':
                uncertainties_epistemic_list.extend(MI)
                uncertainties_aleatoric_list.extend(H_alea)
                uncertainties_list.extend(H_tot) # uncertainty column willl be H_total

            if uncertainty_method == 'MSP':
                uncertainties_list.extend(uncertainty_scores)
            if uncertainty_method == 'MSP_tuned':
                uncertainties_list.extend(uncertainty_scores)
            if uncertainty_method == 'PredictiveEntropy':
                uncertainties_list.extend(H_total)
            if uncertainty_method == 'PredictiveEntropy_tuned':
                uncertainties_list.extend(H_total)
            if uncertainty_method == 'PredictiveEntropy_tsallis':
                uncertainties_list.extend(H_total)   
            if uncertainty_method == 'PredictiveEntropy_tsallis_tuned':
                uncertainties_list.extend(H_total)   

    # end else
    # Save results
    
    df_test["prediction"] = predictions    
    df_test['prob_class_1'] = probabilities_list
    if uncertainty_method not in ['MSP', 'None']:
        df_test['uncertainty_class_1'] = uncertainties_list_class 
        print(uncertainty_method)

    if uncertainty_method == 'DeepEnsemble':
        df_test['uncertainty_epistemic'] = uncertainties_epistemic_list
        df_test['uncertainty_aleatoric'] = uncertainties_aleatoric_list
        df_test['uncertainty'] = uncertainties_list
    if uncertainty_method == 'PredictiveEntropy':
        df_test['uncertainty'] = uncertainties_list
    if uncertainty_method == 'PredictiveEntropy_tsallis':
        df_test['uncertainty'] = uncertainties_list
    if  uncertainty_method == 'MSP':
        df_test['uncertainty'] = uncertainties_list        

    df_test['note']=note
    df_test.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print(f"\n[✓] Inference complete. Predictions saved to: {output_csv}")


# Run when file is executed directly
if __name__ == "__main__":

    for uncertainty_method in uncertainty_methods:
        if uncertainty_method not in ['None', 'MSP',
                                     'PredictiveEntropy', 'PredictiveEntropy_tsallis', 
                                     'McDropout', 'TTA', 'DeepEnsemble', 
                                     'MSP_tuned', 'PredictiveEntropy_tuned', 'PredictiveEntropy_tsallis_tuned'
                                     ]:
            raise ExceptionType("Bad uncertainty_method")
        else:
            uncertainty_method_str = uncertainty_method 

        experiment_name = os.path.basename(experiment_path)
        OUTPUT_CSV = os.path.join('./Outputs', experiment_name + "/predictions_" +uncertainty_method_str+ ".csv")
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

        run_inference_with_uncertainty(
            csv_file=CSV_FILE,
            model_name=model_name,
            list_models_path=list_models_path,
            output_csv=OUTPUT_CSV,
            uncertainty_method=uncertainty_method
        )

