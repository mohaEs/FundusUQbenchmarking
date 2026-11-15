
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from utils import CustomPreprocessing
import numpy as np


from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import roc_auc_score
import math





#################################
################ Conformal UQ #################
#################################

def setup_conformal_prediction(models, cal_loader, alpha=0.1, device='cuda'):
    """Setup conformal prediction threshold"""
    all_scores = []
    
    with torch.no_grad():
        for images, labels in cal_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions using your ensemble
            if len(models) > 1:
                mean_probs, _, _ = ensemble_inference(models, images)
            else:
                outputs = models[0](images)
                mean_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Compute nonconformity scores
            for i, true_label in enumerate(labels.cpu()):
                true_class_prob = mean_probs[i, true_label]
                score = 1 - true_class_prob
                all_scores.append(score)
    
    # Compute threshold
    n = len(all_scores)
    q_level = np.ceil((n+1)*(1-alpha))/n
    threshold = np.quantile(all_scores, q_level)
    
    return threshold

def conformal_predict_batch(mean_probs, threshold):
    """
    Apply conformal prediction to a batch of probabilities
    
    Args:
        mean_probs: numpy array of shape (batch_size, num_classes)
        threshold: conformal threshold
    
    Returns:
        prediction_sets: list of lists containing prediction sets
        mean_probs: same as input (for consistency)
    """
    prediction_sets = []
    
    for i in range(len(mean_probs)):
        # Compute nonconformity scores for each class
        scores = 1 - mean_probs[i]
        
        # Include classes where score <= threshold
        pred_set = np.where(scores <= threshold)[0].tolist()
        prediction_sets.append(pred_set)
    
    return prediction_sets, mean_probs


def interpret_conformal_uncertainty(set_size, num_classes=2):
    if set_size == 1:
        return "High confidence - single diagnosis"
    elif set_size == num_classes:
        return "Maximum uncertainty - all diagnoses possible"
    else:
        return f"Moderate uncertainty - {set_size} possible diagnoses"

#################################
################ Compute optimum calibration values #################
#################################

from utils_metrics import compute_ece
from utils_metrics import compute_uncertainty_auroc

"""
# Full analysis with detailed results
results = find_optimal_parameters(
    model=your_trained_model,
    cal_loader=your_calibration_dataloader,
    temperature_range=(0.1, 10.0),  # Search range for temperature
    q_range=(0.1, 5.0),             # Search range for Tsallis q
    objective='ece',                # 'ece' for calibration, 'auroc' for uncertainty discrimination
    device='cuda'
)

# Access detailed results
print("MSP results:", results['MSP'])
print("Entropy results:", results['Predictive_Entropy']) 
print("Tsallis results:", results['Tsallis'])
print("Evaluation metrics:", results['evaluation'])


"""



def find_optimal_parameters(model, cal_loader, 
                            UQmethod,
                          temperature_range=(0.1, 10.0),
                          q_range=(0.1, 5.0),
                          objective='ece',
                          device='cuda'):
    """
    Find optimal parameters for MSP, Predictive Entropy, and Tsallis methods
    
    Args:
        model: Trained PyTorch model
        cal_loader: Calibration dataloader
        UQmethod: in 'MSP', 'PredictiveEntropy', 'PredictiveEntropy_tsallis'
        temperature_range: Range for temperature search (min, max)
        q_range: Range for Tsallis q parameter search (min, max)
        objective: 'ece' for calibration, 'auroc' for uncertainty discrimination
        device: Device to use for computation
    
    Returns:
        dict with optimal parameters for each method
    """
    model.eval()
    model = model.to(device)
    
    # Collect all calibration data
    print("Collecting calibration data...")
    all_images = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(cal_loader):
            images = images.to(device)
            labels = labels.to(device)
            all_images.append(images)
            all_labels.append(labels)
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Calibration set size: {len(all_images)} samples")
    
    results = {}    
    evaluation_results = {}
    
    
    ################################
    ################################
    ################################
    if UQmethod == 'MSP':
        # 1. Optimize MSP temperature
        print("\n1. Optimizing MSP temperature...")
        
        def msp_objective(temperature):
            probs, uncertainty, _ = msp_inference(
                model, all_images, temperature=temperature, return_numpy=False
            )
            
            if objective == 'ece':
                return compute_ece(probs, all_labels)
            elif objective == 'auroc':
                auroc = compute_uncertainty_auroc(uncertainty, probs, all_labels)
                return -auroc  # Minimize negative AUROC
            else:
                raise ValueError("objective must be 'ece' or 'auroc'")
        
        
        probs_msp, unc_msp, _ = msp_inference(
            model, all_images, 
            temperature=1, 
            return_numpy=True
        )
        evaluation_results['MSP_baseline'] = {
            'ECE': compute_ece(probs_msp, all_labels.cpu().numpy()),
            'AUROC': compute_uncertainty_auroc(unc_msp, probs_msp, all_labels.cpu().numpy())
        }
        
        
        msp_result = minimize_scalar(msp_objective, bounds=temperature_range, method='bounded')
        results['MSP'] = {
            'optimal_temperature': msp_result.x,
            'objective_value': msp_result.fun,
            'success': msp_result.success
        }
        
        
        # MSP
        probs_msp, unc_msp, _ = msp_inference(
            model, all_images, 
            temperature=results['MSP']['optimal_temperature'], 
            return_numpy=True
        )
        evaluation_results['MSP_tuned'] = {
            'ECE': compute_ece(probs_msp, all_labels.cpu().numpy()),
            'AUROC': compute_uncertainty_auroc(unc_msp, probs_msp, all_labels.cpu().numpy())
        }
        
    
    
    ################################
    ################################
    ################################
    if UQmethod == 'PredictiveEntropy' or UQmethod=='PredictiveEntropy_tsallis':    
        # 2. Optimize Predictive Entropy temperature
        print("\n2. Optimizing Predictive Entropy temperature...")
        
        def entropy_objective(temperature):
            probs, uncertainty, _, _ = predictive_entropy_inference(
                model, all_images, temperature=temperature, return_numpy=False
            )
            
            if objective == 'ece':
                return compute_ece(probs, all_labels)
            elif objective == 'auroc':
                auroc = compute_uncertainty_auroc(uncertainty, probs, all_labels)
                return -auroc
            else:
                raise ValueError("objective must be 'ece' or 'auroc'")
        
        # Predictive Entropy
        probs_ent, unc_ent, _, _ = predictive_entropy_inference(
            model, all_images,
            temperature=1,
            return_numpy=True
        )
        evaluation_results['PredictiveEntropy_baseline'] = {
            'ECE': compute_ece(probs_ent, all_labels.cpu().numpy()),
            'AUROC': compute_uncertainty_auroc(unc_ent, probs_ent, all_labels.cpu().numpy())
        }
        
        
        entropy_result = minimize_scalar(entropy_objective, bounds=temperature_range, method='bounded')
        results['PredictiveEntropy'] = {
            'optimal_temperature': entropy_result.x,
            'objective_value': entropy_result.fun,
            'success': entropy_result.success
        }
        
        
        # Predictive Entropy
        probs_ent, unc_ent, _, _ = predictive_entropy_inference(
            model, all_images,
            temperature=results['PredictiveEntropy']['optimal_temperature'],
            return_numpy=True
        )
        evaluation_results['PredictiveEntropy_tuned'] = {
            'ECE': compute_ece(probs_ent, all_labels.cpu().numpy()),
            'AUROC': compute_uncertainty_auroc(unc_ent, probs_ent, all_labels.cpu().numpy())
        }
        

    
    
    ################################
    ################################
    ################################
    
    # if UQmethod=='PredictiveEntropy_tsallis':    
    #     print("\n3. Debugging Tsallis q parameter...")
    #     optimal_temp = results['PredictiveEntropy']['optimal_temperature']
        
        # # Deep debugging: Check if q actually changes the uncertainty values
        # print("   Detailed debugging of q parameter effects...")
        
        # debug_q_values = [0.5, 1.2, 2.0, 3.0]
        
        # for q in debug_q_values:
        #     probs, uncertainty, _, _ = predictive_tsallis_entropy_inference(
        #         model, all_images, q=q, temperature=optimal_temp, return_numpy=True
        #     )
            
        #     # Check if probabilities are identical (they should be)
        #     if q == debug_q_values[0]:
        #         first_probs = probs.copy()
        #         first_uncertainty = uncertainty.copy()
            
        #     # Compare probabilities
        #     probs_identical = np.allclose(probs, first_probs, atol=1e-8)
        #     unc_identical = np.allclose(uncertainty, first_uncertainty, atol=1e-8)
            
        #     print(f"    q={q:.1f}:")
        #     print(f"      Probs identical to q=0.5: {probs_identical}")
        #     print(f"      Uncertainty identical to q=0.5: {unc_identical}")
        #     print(f"      Uncertainty: min={uncertainty.min():.6f}, max={uncertainty.max():.6f}, mean={uncertainty.mean():.6f}")
        #     print(f"      Uncertainty std: {uncertainty.std():.6f}")
            
        #     # Check a few specific values
        #     print(f"      First 5 uncertainty values: {uncertainty[:5]}")
        #     print()
        
        # # If uncertainties are identical, there's a bug in the Tsallis implementation
        # if np.allclose(first_uncertainty, uncertainty, atol=1e-8):
        #     print("   ERROR: Tsallis uncertainties are identical across q values!")
        #     print("   This indicates a bug in predictive_tsallis_entropy_inference function")
            
        #     # Test the Tsallis function directly
        #     print("\n   Testing Tsallis implementation with simple inputs...")
            
        #     # Create simple test case
        #     test_logits = torch.tensor([[2.0, 1.0, 0.5], [1.0, 2.0, 0.3]])
            
        #     for q in [0.5, 1.2, 2.0]:
        #         probs = torch.softmax(test_logits / optimal_temp, dim=1)
        #         p = probs.clamp(min=1e-12, max=1 - 1e-12)
                
        #         if abs(q - 1.0) < 1e-6:
        #             # Shannon entropy
        #             H_total = -(p * p.log()).sum(dim=1)
        #         else:
        #             # Tsallis entropy
        #             p_q = p.pow(q)
        #             sum_pq = p_q.sum(dim=1)
        #             H_total = (1.0 - sum_pq) / (q - 1.0)
                
        #         print(f"    Direct computation q={q}: {H_total.numpy()}")
        
        # else:
        #     print("   Tsallis uncertainties DO vary with q - problem is elsewhere")
            
        #     # Check if the AUROC computation is the issue
        #     print("\n   Testing AUROC computation...")
            
        #     for q in [0.5, 2.0]:
        #         probs, uncertainty, _, _ = predictive_tsallis_entropy_inference(
        #             model, all_images, q=q, temperature=optimal_temp, return_numpy=True
        #         )
                
        #         # Manual AUROC calculation
        #         predicted_classes = np.argmax(probs, axis=1)
        #         incorrect_mask = (predicted_classes != all_labels.cpu().numpy())
                
        #         print(f"    q={q}: incorrect_predictions={incorrect_mask.sum()}/{len(incorrect_mask)}")
                
        #         if incorrect_mask.sum() > 0 and incorrect_mask.sum() < len(incorrect_mask):
        #             auroc = roc_auc_score(incorrect_mask, uncertainty)
        #             print(f"    q={q}: manual AUROC={auroc:.6f}")
    
    if UQmethod=='PredictiveEntropy_tsallis':    
        # 3. Optimize Tsallis q 
        print("\n3. Optimizing Tsallis q ... ")
        optimal_temp = results['PredictiveEntropy']['optimal_temperature']
        #results['PredictiveEntropy_tsallis']['optimal_temperature'] = optimal_temp  # Store for reference

        # Now optimize only q parameter using the fixed temperature
        def tsallis_q_objective(q):
            # Skip problematic q values near 1.0
            if abs(q - 1.0) < 0.05:
                return 1e6
                
            try:
                probs, uncertainty, _, _ = predictive_tsallis_entropy_inference(
                    model, all_images, 
                    q=q, 
                    temperature=optimal_temp,  # Fixed temperature
                    return_numpy=False
                )
                
                score = compute_uncertainty_auroc(uncertainty, probs, all_labels)
                score = -score  # Minimize negative AUROC
                
                return score
                    
            except Exception as e:
                print(f"    Error with q={q:.3f}: {e}")
                return 1e6


        # # Add this to your debugging section after computing uncertainties for different q values:
        
        # print("\n   Testing uncertainty rankings across q values...")
        
        # # Store uncertainties for comparison
        # uncertainties_dict = {}
        # for q in [0.5, 1.2, 2.0, 3.0]:
        #     probs, uncertainty, _, _ = predictive_tsallis_entropy_inference(
        #         model, all_images, q=q, temperature=optimal_temp, return_numpy=True
        #     )
        #     uncertainties_dict[q] = uncertainty.copy()
        
        # # Check rank correlations between different q values
        # from scipy.stats import spearmanr
        # print("   Rank correlations between q values:")
        # q_values = [0.5, 1.2, 2.0, 3.0]
        # for i, q1 in enumerate(q_values):
        #     for q2 in q_values[i+1:]:
        #         corr = spearmanr(uncertainties_dict[q1], uncertainties_dict[q2])[0]
        #         print(f"     q={q1} vs q={q2}: {corr:.6f}")
        
        # # Check if the same samples are ranked as most/least uncertain
        # n_check = 10
        # print(f"\n   Top {n_check} most uncertain samples by index:")
        # for q in q_values:
        #     top_indices = np.argsort(uncertainties_dict[q])[-n_check:]
        #     print(f"     q={q}: {top_indices}")
        
        # print(f"\n   Bottom {n_check} least uncertain samples by index:")
        # for q in q_values:
        #     bottom_indices = np.argsort(uncertainties_dict[q])[:n_check]
        #     print(f"     q={q}: {bottom_indices}")
        
        # # Additional analysis: Check the actual error prediction capability
        # predicted_classes = np.argmax(probs, axis=1)
        # all_labels_np = all_labels.cpu().numpy()
        # incorrect_mask = (predicted_classes != all_labels_np)
        
        # print(f"\n   Model performance analysis:")
        # print(f"     Accuracy: {(~incorrect_mask).mean():.3f}")
        # print(f"     Errors: {incorrect_mask.sum()}/{len(incorrect_mask)}")
        
        # # Check uncertainty distribution for correct vs incorrect predictions
        # for q in [0.5, 2.0, 3.0]:
        #     unc = uncertainties_dict[q]
        #     correct_unc = unc[~incorrect_mask]
        #     incorrect_unc = unc[incorrect_mask]
            
        #     print(f"     q={q}: Correct predictions uncertainty mean={correct_unc.mean():.4f}")
        #     print(f"     q={q}: Incorrect predictions uncertainty mean={incorrect_unc.mean():.4f}")
        #     print(f"     q={q}: Difference: {incorrect_unc.mean() - correct_unc.mean():.4f}")


            
        # Grid search for q only
        print("   Performing grid search for q parameter (optimizing uncertainty AUROC)...")
        q_values = [0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
        
        best_q = 2.0
        best_score = float('inf')
        
        for q in q_values:
            score = tsallis_q_objective(q)
            # Convert back to positive AUROC for display
            auroc_display = -score if score != 1e6 else 0.0
            print(f"    q={q:.1f}: uncertainty_AUROC={auroc_display:.6f}")
            
            if score < best_score:
                best_score = score
                best_q = q
        
        print(f"   Optimal Tsallis q: {best_q:.3f}")
        print(f"   Using temperature: {optimal_temp:.3f}")
        print(f"   Best uncertainty AUROC: {-best_score:.4f}")
        
        # Store results
        results['PredictiveEntropy_tsallis'] = {
            'optimal_q': best_q,
            'optimal_temperature': optimal_temp,  # Reused from Predictive Entropy
            'objective_value': best_score,
            'success': True
        }
        
        # Baseline evaluation (Shannon entropy with same temperature)
        probs_baseline, unc_baseline, _, _ = predictive_tsallis_entropy_inference(
            model, all_images, q=0.9, temperature=optimal_temp, return_numpy=True
        )
        evaluation_results['PredictiveEntropy_tsallis_baseline'] = {
            'ECE': compute_ece(probs_baseline, all_labels.cpu().numpy()),
            'AUROC': compute_uncertainty_auroc(unc_baseline, probs_baseline, all_labels.cpu().numpy())
        }
        
        # Optimized Tsallis evaluation
        probs_tsallis, unc_tsallis, _, _ = predictive_tsallis_entropy_inference(
            model, all_images,
            q=best_q,
            temperature=optimal_temp,
            return_numpy=True
        )
        evaluation_results['PredictiveEntropy_tsallis_tuned'] = {
            'ECE': compute_ece(probs_tsallis, all_labels.cpu().numpy()),
            'AUROC': compute_uncertainty_auroc(unc_tsallis, probs_tsallis, all_labels.cpu().numpy())
        }
        

    
    ########################
    ########################
    ########################
    
    # Print summary
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION SUMMARY")    
    print(f"Optimization objective: {objective.upper()}")
    print(f"Calibration set size: {len(all_images)} samples")
       
    results['evaluation'] = evaluation_results
    results['optimization_info'] = {
        'objective': objective,
        'calibration_size': len(all_images),
        'temperature_range': temperature_range,
        'q_range': q_range,
        'device': device
    }
    
    return results






#################################
################ MSP #################
#################################


@torch.no_grad()
def msp_inference(
    model,
    images,
    temperature: float = 1.0,   # plug in your calibrated T
    return_numpy: bool = True,
):
    """
    MSP-based uncertainty (single forward pass; post-hoc).

    Inputs
    ------
    model : torch.nn.Module (will be set to eval())
    images: Tensor (B, ...) on the correct device
    temperature: float, optional temperature scaling for logits
    return_numpy: if True, convert outputs to numpy arrays

    Returns
    -------
    dict with:
      logits           : (B, C)
      probs            : (B, C)     # softmax(logits/T)
      pred_class       : (B,)       # argmax over classes (long)
      msp              : (B,)       # max softmax probability (confidence), in [0,1]
      uncertainty_msp  : (B,)       # 1 - msp, in [0,1]
      margin           : (B,)       # p_top1 - p_top2, in [0,1]; optional extra
    Notes
    -----
    - Larger `msp` = more confident; larger `uncertainty_msp` = less confident.
    - `margin` is a useful companion score (bigger margin => clearer separation).
    """
    model.eval()
    logits = model(images)                                # (B, C)
    probs  = torch.softmax(logits / temperature, dim=1)   # (B, C)

    msp_vals, pred_cls = probs.max(dim=1)                 # (B,), (B,)
    uncertainty = 1.0 - msp_vals

    # Optional: margin between top-1 and top-2 probabilities
    top2 = torch.topk(probs, k=2, dim=1).values           # (B, 2)
    margin = top2[:, 0] - top2[:, 1]                      # (B,)

    out = {
        "logits": logits,
        "probs": probs,
        "pred_class": pred_cls,
        "msp": msp_vals,
        "uncertainty_msp": uncertainty,
        "margin": margin,
    }

    if return_numpy:
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
    
    note = f"MSP inference with temperature={temperature}"
    
    return out["probs"], out["uncertainty_msp"], note



#################################
#################################
#################################


# Define TTA transforms
def get_tta_transforms(image_size, num_tta_samples=10):
    """Generate different augmented versions for Test Time Augmentation"""
    
    # Your custom random transform
    random_transform = transforms.Compose([
        CustomPreprocessing(),  # Commented out as in your example
        transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),  # Resize slightly larger
        transforms.RandomCrop((image_size, image_size)),  # Then random crop to final size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create a list with the same transform repeated num_tta_samples times
    # Each call will generate different random augmentations due to the random nature of transforms
    tta_transforms = [random_transform for _ in range(num_tta_samples)]
    
    return tta_transforms


# Custom dataset for TTA
class TTAInferenceDataset(Dataset):
    def __init__(self, dataframe, tta_transforms):
        self.dataframe = dataframe
        self.tta_transforms = tta_transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        
        # Apply all TTA transforms
        augmented_images = []
        for transform in self.tta_transforms:
            augmented_images.append(transform(image))
        
        return torch.stack(augmented_images), img_path

def tta_inference(model, augmented_images):
    """Perform Test Time Augmentation inference"""
    
    
    batch_size, num_augs, channels, height, width = augmented_images.shape
    # Reshape to process all augmentations at once
    all_images = augmented_images.view(-1, channels, height, width)
    
    with torch.no_grad():
        outputs = model(all_images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Reshape back and average predictions
    probabilities = probabilities.view(batch_size, num_augs, -1)
    mean_pred = torch.mean(probabilities, dim=1).cpu().numpy()
    uncertainty = torch.std(probabilities, dim=1).cpu().numpy()
    
    return mean_pred, uncertainty


#################################
#################################
#################################

def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout"""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def mc_dropout_inference(model, images, num_samples=50):
    """Perform MC Dropout inference"""
    enable_dropout(model)
    
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions.append(probabilities.cpu().numpy())
    
    predictions = np.array(predictions)  # Shape: (num_samples, batch_size, num_classes)
    
    # Calculate mean and uncertainty
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    
    note= f"MC Dropout inference with {num_samples} forward passes"
    return mean_pred, uncertainty, note



#####################################
#####################################
#####################################


import math

@torch.no_grad()
def predictive_entropy_inference(
    model,
    images,
    temperature: float = 1.0,
    eps: float = 1e-12,
    return_numpy: bool = True,
):
    """
    Shannon entropy inference (single forward pass).
    Single-pass predictive entropy (no MC Dropout).

    Inputs
    ------
    model : torch.nn.Module (ideally model.eval())
    images: Tensor (B, ...) on the correct device
    temperature: float, optional temperature scaling (T=1 means no change)
    eps   : numerical stability for logs
    return_numpy: if True, return numpy arrays; else torch tensors

    Returns
    -------
    dict with:
      probs                  : (B, C)
      H_total_nats           : (B,)     # predictive entropy in nats
      H_total_norm           : (B,)     # normalized to [0,1] via ln(C)
      H_ovr_nats             : (B, C)   # one-vs-rest entropy per class in nats
      H_ovr_norm             : (B, C)   # normalized to [0,1] via ln(2)
      classwise_contrib_nats : (B, C)   # c_k = -p_k * log p_k
      classwise_share        : (B, C)   # c_k / H(p); sums to 1 when H>0

    Notes
    -----
    - In binary classification the two classwise one-vs-rest values are identical and each equals the total—exactly what you printed.
    - For binary C=2: H_ovr_nats[:,0] == H_ovr_nats[:,1] == H_total_nats.
    - When H_total_nats is ~0 (near one-hot), classwise_share is set near 0
      (denominator clamped to eps).
    - All logs are natural (nats). Normalize to [0,1] as provided.
    """
    logits = model(images)                      # (B, C)
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=1)        # (B, C)
    p = probs.clamp(min=eps, max=1 - eps)

    # Total predictive entropy H(p)
    H_total_nats = -(p * p.log()).sum(dim=1)    # (B,)
    B, C = p.shape
    lnC = math.log(C)
    H_total_norm = H_total_nats / max(lnC, eps) # (B,) in [0,1]

    # One-vs-rest entropy per class h(p_k)
    H_ovr_nats = -(p * p.log() + (1 - p) * (1 - p).clamp_min(eps).log())  # (B, C)
    ln2 = math.log(2.0)
    H_ovr_norm = H_ovr_nats / ln2                 # (B, C) each in [0,1]

    # Per-class contributions and shares
    classwise_contrib_nats = -(p * p.log())       # (B, C)
    denom = H_total_nats.unsqueeze(1).clamp_min(eps)
    classwise_share = classwise_contrib_nats / denom  # (B, C)

    out = {
        "probs": probs,
        "H_total_nats": H_total_nats,
        "H_total_norm": H_total_norm,
        "H_ovr_nats": H_ovr_nats,
        "H_ovr_norm": H_ovr_norm,
        "classwise_contrib_nats": classwise_contrib_nats,
        "classwise_share": classwise_share,
    }

    if return_numpy:
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
            
    note = f"Predictive entropy inference with temperature={temperature}, eps={eps}"
    return out["probs"], out["H_total_nats"], out["H_ovr_nats"], note


#####################################
#####################################
#####################################


import math

def _ln_q(x: float, q: float, eps: float = 1e-12) -> float:
    if abs(q - 1.0) < 1e-6:
        return math.log(max(x, eps))
    return (x**(1.0 - q) - 1.0) / (1.0 - q)

@torch.no_grad()
def predictive_tsallis_entropy_inference(
    model,
    images,
    q: float = 1.1,              # choose q>0, q!=1; q≈1 → Shannon
    temperature: float = 1.0,    # plug your calibrated T here
    eps: float = 1e-12,
    return_numpy: bool = True,
):
    """
    Per-sample predictive uncertainty via Tsallis entropy (single forward pass).

    Returns dict with:
      probs                    : (B,C)
      Hq_total                 : (B,)     # Tsallis H_q(p)
      Hq_total_norm            : (B,)     # normalized by ln_q(C) to [0,1]
      Hq_ovr                   : (B,C)    # one-vs-rest Tsallis h_q(p_k)
      Hq_ovr_norm              : (B,C)    # normalized by ln_q(2) to [0,1]
      classwise_contrib_q      : (B,C)    # (p_k - p_k^q)/(q-1), sums to Hq_total
      classwise_share_q        : (B,C)    # contribution fraction per class
    Notes:
        Pick q on a validation set to optimize what you care about (e.g., error detection AUROC, AURC for risk–coverage, or OOD AUROC if you include cross-device samples).
        If your minority disease class suffers from over-referral, try q>1 (less sensitive to tiny tail probs).
        If you want to flag diffuse/ambiguous predictions more aggressively (e.g., cross-camera variation), try q<1.
        If minority class is penalized by too many tiny alternative probabilities → try q > 1.
        If you want to be extra sensitive to “diffuse” predictions (possible cross-camera shift/OOD) → try q < 1 and validate that minority-class error detection doesn’t degrade.

      - If q≈1, this reduces to Shannon entropy and one-vs-rest binary entropy.
      - For binary C=2, Hq_ovr[:,0] == Hq_ovr[:,1] == Hq_total (expected).
    """
    assert q > 0, "q must be > 0"
    logits = model(images)
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=1)            # (B,C)
    p = probs.clamp(min=eps, max=1 - eps)
    B, C = p.shape

    if abs(q - 1.0) < 1e-6:
        # Shannon limit
        H_total = -(p * p.log()).sum(dim=1)         # (B,)
        H_ovr   = -(p * p.log() + (1 - p) * (1 - p).clamp_min(eps).log())  # (B,C)
        classwise_contrib_q = -(p * p.log())        # (B,C) (Shannon contributions)
        lnC = math.log(C)
        ln2 = math.log(2.0)
    else:
        # Tsallis H_q(p) = (1 - sum p^q) / (q - 1)
        p_q = p.pow(q)                               # (B,C)
        sum_pq = p_q.sum(dim=1)                      # (B,)
        H_total = (1.0 - sum_pq) / (q - 1.0)         # (B,)

        # One-vs-rest: h_q(p_k) = (1 - (p_k^q + (1-p_k)^q)) / (q - 1)
        one_minus_p = (1 - p).clamp_min(eps)
        H_ovr = (1.0 - (p.pow(q) + one_minus_p.pow(q))) / (q - 1.0)  # (B,C)

        # Per-class contributions that sum to H_total:
        # H_q = sum_k (p_k - p_k^q) / (q - 1)
        classwise_contrib_q = (p - p.pow(q)) / (q - 1.0)  # (B,C)

        lnC = _ln_q(float(C), q)
        ln2 = _ln_q(2.0, q)

    # Normalizations
    Hq_total_norm = H_total / max(lnC, eps)                       # (B,)
    Hq_ovr_norm   = H_ovr / max(ln2, eps)                         # (B,C)

    # Shares
    denom = H_total.unsqueeze(1).clamp_min(eps)                   # (B,1)
    classwise_share_q = classwise_contrib_q / denom               # (B,C)

    out = {
        "probs": probs,
        "Hq_total": H_total,
        "Hq_total_norm": Hq_total_norm,
        "Hq_ovr": H_ovr,
        "Hq_ovr_norm": Hq_ovr_norm,
        "classwise_contrib_q": classwise_contrib_q,
        "classwise_share_q": classwise_share_q,
    }
    if return_numpy:
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
    
    note = f"Predictive entropy inference with q={q}, temperature={temperature}, eps={eps}"

    return out["probs"], out["Hq_total"], out["Hq_ovr"], note



#####################################
#####################################
#####################################


import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.functional as F

def ensemble_inference(models, images):
    """
    Perform deep-ensemble inference with outputs .
    Returns:
        mean_pred: (B, C) numpy array
        uncertainty: (B, C) numpy array  # std across models (classwise)
        proxy of epistemic uncertainty
        simple, similar to MC-Dropout uncertainty is calculated as the standard deviation of predictions across models.
    """
    predictions = []

    # Put all models in eval mode (dropout off)
    for m in models:
        m.eval()

    # Send images to the same device as the first model
    device = next(models[0].parameters()).device
    images = images.to(device, non_blocking=True)

    with torch.no_grad():
        for m in models:
            logits = m(images)                       # [B, C]
            probs  = F.softmax(logits, dim=1)        # [B, C]
            predictions.append(probs.detach().cpu().numpy())

    # Shape: (num_models, B, C)
    predictions = np.asarray(predictions, dtype=np.float32)

    # Mean and classwise std across models -> shapes (B, C)
    mean_pred  = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)  # matches MC-Dropout "uncertainty" shape

    note = f"Deep ensemble inference with {len(models)} models"
    return mean_pred, uncertainty, note


# OPTIONAL: if you also want scalar UQ terms like MI (epistemic) with batch shape (B,)
def ensemble_entropy_terms(models, images):
    """
    Returns:
        H_total:  (B,) predictive entropy of mean probs
        H_alea:   (B,) expected entropy across models (aleatoric)
        MI_epist: (B,) epistemic = H_total - H_alea
    """
    preds = []

    for m in models:
        m.eval()

    device = next(models[0].parameters()).device
    images = images.to(device, non_blocking=True)

    with torch.no_grad():
        for m in models:
            p = F.softmax(m(images), dim=1)
            preds.append(p)  # [B, C] tensors

    # [M, B, C] torch tensor
    probs = torch.stack(preds, dim=0)

    eps = 1e-12
    def entropy(p):  # p: [..., C]
        p = torch.clamp(p, min=eps, max=1.0)
        return -(p * p.log()).sum(dim=-1)

    p_mean = probs.mean(dim=0)           # [B, C]
    H_total = entropy(p_mean)            # [B]
    H_alea  = entropy(probs).mean(dim=0) # [B]
    MI_epist = H_total - H_alea          # [B]

    return H_total.cpu().numpy(), H_alea.cpu().numpy(), MI_epist.cpu().numpy()

