import os
import pandas as pd
import numpy as np

# ---------- Conformal Helpers ----------
def compute_conformal_threshold(cal_probs, cal_labels, alpha=0.1):
    """
    Compute conformal threshold from calibration set.
    cal_probs: array of prob_class_1 for calibration samples
    cal_labels: true labels (0/1) for calibration samples
    """
    # Nonconformity score = 1 - p_true
    scores = np.where(cal_labels == 1, 1 - cal_probs, cal_probs)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.sort(scores)[k - 1]


def conformal_predict_binary(probs, threshold):
    """
    Apply conformal prediction for binary classification.
    probs: array of predicted probabilities for class=1
    threshold: conformal threshold
    Returns: list of prediction sets
    """
    pred_sets = []
    for p in probs:
        # Score for class 0 = prob that true=0 → 1 - p_true = p
        # Score for class 1 = prob that true=1 → 1 - p_true = 1 - p
        scores = [p, 1 - p]
        pred_set = [cls for cls, s in zip([0, 1], scores) if s <= threshold]
        pred_sets.append(pred_set if pred_set else [0, 1])  # avoid empty sets
    return pred_sets


# ---------- Main Processing ----------
def process_directory(csv_dir, alpha=0.1, out_suffix="_WithConformal.csv"):
    for file in os.listdir(csv_dir):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(csv_dir, file)

        if "MSP" in path:
            continue
        if "predictions_PredictiveEntropy" in path:
            continue

        df = pd.read_csv(path)

        # Split calibration vs test
        cal_df = df[df["tag_4calibrationSet"] == 1].copy()
        test_df = df[df["tag_4calibrationSet"] == 0].copy()

        cal_probs = cal_df["prob_class_1"].values
        cal_labels = cal_df["prediction"].values.astype(int)

        # Threshold
        threshold = compute_conformal_threshold(cal_probs, cal_labels, alpha)

        # Apply to test set
        test_probs = test_df["prob_class_1"].values
        pred_sets = conformal_predict_binary(test_probs, threshold)

        # Add results to test set
        test_df["conf_pred_set"] = pred_sets
        test_df["set_sizes"] = test_df["conf_pred_set"].apply(len)
        test_df["true_label_in_set"] = [
            int(true in pred_set) for true, pred_set in zip(test_df["prediction"], pred_sets)
            ]
        test_df["isUncertain"] = test_df["set_sizes"] - 1  # 0 = confident, 1 = uncertain

        test_df["Conformal Threshold"] = threshold
        test_df["note_conformal"] = "Conformal prediction applied with alpha=" + str(alpha)
        
        # Merge back
        out_df = pd.concat([cal_df, test_df]).sort_index()

        # Save
        out_path = os.path.join(csv_dir, file.replace(".csv", out_suffix))
        out_df.to_csv(out_path, index=False)

        # Print summary
        coverage = test_df["true_label_in_set"].mean()
        avg_size = test_df["set_sizes"].mean()
        print(f"\nFile: {file}")
        print(f"  Coverage (@1-alpha): {coverage:.3f}")
        print(f"  Avg set size:        {avg_size:.3f}")
        print(f"  Threshold:           {threshold:.4f}")


# ---------- Run ----------

if __name__ == "__main__":

    path_savedPredictions = "./Outputs/FocalLoss/GL_ViT_ensemble_1"  # Directory containing CSV files
    process_directory(path_savedPredictions, alpha=0.1, out_suffix="_conformal.csv")