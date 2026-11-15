# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 14:45:09 2025

@author: mohae
"""


import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import roc_auc_score
import math

def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error"""
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_probs = probs.max(dim=1)[0]
    predicted_classes = probs.argmax(dim=1)
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower.item()) & (max_probs <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = (predicted_classes[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def compute_uncertainty_auroc(uncertainties, probs, labels):
    """Compute AUROC for uncertainty predicting model errors"""
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if isinstance(uncertainties, np.ndarray):
        uncertainties = torch.from_numpy(uncertainties)
    
    predicted_classes = probs.argmax(dim=1)
    incorrect_mask = (predicted_classes != labels).cpu().numpy()
    
    if incorrect_mask.sum() == 0 or incorrect_mask.sum() == len(incorrect_mask):
        return 0.5  # No errors or all errors
    
    try:
        auroc = roc_auc_score(incorrect_mask, uncertainties.cpu().numpy())
        return auroc
    except:
        return 0.5
    
    
    