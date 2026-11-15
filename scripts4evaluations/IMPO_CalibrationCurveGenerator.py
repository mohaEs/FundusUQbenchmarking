import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import numpy as np
import os

path_save = './figures_tables'
os.makedirs(path_save, exist_ok=True)

# --- Configuration ---
DISEASE_MAP = {
    'AMD': 'AMD',
    'DR': 'Diabetes',
    'GL': 'Glaucoma'
}

# Define the methods and corresponding files we want to plot
METHODS_TO_PLOT = {
    'Baseline': ('ensemble_1', 'predictions_None.csv'),
    'MC Dropout': ('ensemble_1', 'predictions_McDropout.csv'),
    'TTA': ('ensemble_1', 'predictions_TTA.csv'),
    'Deep Ensemble': ('ensemble_1', 'predictions_DeepEnsemble.csv')
}

MODELS = ['ViT', 'RETFoundGreen', 'ViT(FLSD-53)']
BASE_OUTPUT_PATH_CELoss = Path('../Outputs/CELoss')
BASE_OUTPUT_PATH_FocalLoss = Path('../Outputs/FocalLoss')


def load_calibration_data():
    """
    Scans the 'Outputs' directory to load data for calibration curves,
    filtering out rows where 'tag_4calibrationSet' is 1, and reports the count.
    """
    all_data = []
    total_removed_count = 0
    print("Starting data loading for calibration curves...")

    for disease_code, disease_name in DISEASE_MAP.items():
        for method_name, (folder_suffix, filename) in METHODS_TO_PLOT.items():
            for model in MODELS:
                if model=='ViT(FLSD-53)':
                    model0 = 'ViT'
                    folder_name = f"{disease_code}_{model0}_{folder_suffix}"
                    file_path = BASE_OUTPUT_PATH_FocalLoss / folder_name / filename
                
                else:
                    folder_name = f"{disease_code}_{model}_{folder_suffix}"
                    file_path = BASE_OUTPUT_PATH_CELoss / folder_name / filename

                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Filter out rows where 'tag_4calibrationSet' is 1
                        if 'tag_4calibrationSet' in df.columns:
                            initial_rows = len(df)
                            df = df[df['tag_4calibrationSet'] != 1].copy()
                            removed_rows = initial_rows - len(df)
                            total_removed_count += removed_rows
                            if removed_rows > 0:
                                print(f"Filtered {removed_rows} calibration rows from {file_path}")
                        
                        # Ensure required columns exist after filtering
                        if 'label' in df.columns and 'prob_class_1' in df.columns:
                            temp_df = pd.DataFrame({
                                'label': df['label'],
                                'prob_class_1': df['prob_class_1'],
                                'Method': method_name,
                                'Model': model,
                                'Disease': disease_name
                            })
                            all_data.append(temp_df)
                        else:
                            print(f"Columns 'label' or 'prob_class_1' not found in {file_path}, skipping.")
                    except Exception as e:
                        print(f"Could not process {file_path}. Error: {e}")
                else:
                    print(f"File not found, skipping: {file_path}")

    if not all_data:
        print("\nERROR: No data was loaded.")
        return pd.DataFrame()

    print(f"\nTotal calibration rows dropped across all files: {total_removed_count}")
    print("Data loading complete.")
    return pd.concat(all_data, ignore_index=True)

import matplotlib as mpl

# --- set font sizes globally ---
mpl.rcParams.update({
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
})

def generate_calibration_figure(data_df):
    """
    Generates and saves a 2x3 figure of reliability diagrams (calibration curves)
    for all models and diseases.
    """
    print("\nGenerating calibration curve figure...")

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(24, 16), sharex=True, sharey=True)
    # fig.suptitle('Model Calibration Analysis') # Title removed as requested

    # Use the same professional color palette as the other script
    method_colors = sns.color_palette("colorblind", len(METHODS_TO_PLOT))
    color_map = {method: color for method, color in zip(METHODS_TO_PLOT.keys(), method_colors)}

    for row_idx, model in enumerate(MODELS):
        for col_idx, disease_name in enumerate(DISEASE_MAP.values()):
            ax = axes[row_idx, col_idx]
            
            # Create a secondary y-axis for the histogram
            ax_hist = ax.twinx()
            ax_hist.set_yticks([])  # Hide histogram y-axis ticks

            for method in METHODS_TO_PLOT.keys():
                subset = data_df[
                    (data_df['Model'] == model) & 
                    (data_df['Disease'] == disease_name) & 
                    (data_df['Method'] == method)
                ]
                
                if not subset.empty:
                    y_true = subset['label']
                    y_prob = subset['prob_class_1']
                    
                    # Calibration curve
                    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
                    ax.plot(mean_pred, frac_pos, "s-",
                            label=method, color=color_map[method], lw=2.5)
                    
                    # Histogram
                    ax_hist.hist(y_prob, range=(0, 1), bins=10, color=color_map[method],
                                 histtype="step", lw=2, alpha=0.3)

            # Perfect calibration line
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            
            # Titles and labels (no fontsize needed—controlled by rcParams)
            if row_idx == 0:
                ax.set_title(disease_name)
            if col_idx == 0:
                ax.set_ylabel(f'{model}\nCorrectness')
            if row_idx == 1:
                ax.set_xlabel('Mean Predicted Probability (Confidence)')
            
            ax.grid(True)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

    # Shared legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))

    # Layout adjustment
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    output_filename = os.path.join(path_save, 'Model_Calibration_Curves.pdf')
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved calibration figure: {output_filename}")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE) of a model.
    """
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_limits[1:-1])
    
    ece = 0
    for bin_i in range(n_bins):
        in_bin = bin_indices == bin_i
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
    return ece


def generate_calibration_metrics_table(data_df):
    """
    Calculates NLL, ECE, and Brier Score, and exports them as a PNG image
    and a CSV file, formatted like the AUROC/AUPRC tables.
    """
    print("\nGenerating calibration metrics table...")
    
    metrics = []
    # Calculate scores for each combination
    for disease in data_df['Disease'].unique():
        for model in data_df['Model'].unique():
            for method in data_df['Method'].unique():
                subset = data_df[(data_df['Disease'] == disease) & 
                                 (data_df['Model'] == model) & 
                                 (data_df['Method'] == method)]
                if not subset.empty:
                    y_true = subset['label']
                    y_prob = subset['prob_class_1']
                    
                    nll = log_loss(y_true, y_prob)
                    ece = expected_calibration_error(y_true, y_prob)
                    brier = brier_score_loss(y_true, y_prob)
                    
                    metrics.append({
                        'Disease': disease, 'Model': model, 'Method': method,
                        'NLL': nll, 'ECE': ece, 'Brier': brier
                    })

    metrics_df = pd.DataFrame(metrics)
    
    # --- Create separate pivot tables for each metric ---
    nll_pivot = metrics_df.pivot_table(index='Disease', columns=['Model', 'Method'], values='NLL')
    ece_pivot = metrics_df.pivot_table(index='Disease', columns=['Model', 'Method'], values='ECE')
    brier_pivot = metrics_df.pivot_table(index='Disease', columns=['Model', 'Method'], values='Brier')
    
    # --- Define consistent ordering ---
    model_method_order = [(model, method) for model in MODELS for method in METHODS_TO_PLOT.keys()]
    disease_order = [DISEASE_MAP[code] for code in ['AMD', 'DR', 'GL']]
    
    # Apply ordering to all pivots
    pivots = [nll_pivot, ece_pivot, brier_pivot]
    pivots = [p.reindex(index=disease_order, columns=model_method_order) for p in pivots]

    # --- Save combined table to CSV file ---
    combined_pivot = pd.concat(pivots, keys=['NLL', 'ECE', 'Brier'])
    csv_filename = os.path.join(path_save, 'calibration_metrics.csv')
    combined_pivot.to_csv(csv_filename)
    print(f"Saved combined calibration metrics to: {csv_filename}")

    # # --- Plotting the tables as a PNG ---
    # fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    # fig.patch.set_facecolor('white')

    # titles = ['NLL', 'ECE', 'Brier']
    
    # for ax, pivot_table, title in zip(axes, pivots, titles):
    #     ax.axis('off')
        
    #     # --- Prepare data for a stable table structure ---
    #     plot_df = pivot_table.round(3).reset_index()
        
    #     col_labels = [''] + [f'{model}\n{method}' for model, method in pivot_table.columns]
        
    #     col_widths = [0.09] + [0.125] * 8

    #     table = ax.table(cellText=plot_df.values,
    #                      colLabels=col_labels,
    #                      loc='center',
    #                      cellLoc='center',
    #                      colWidths=col_widths)
        
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(11)
    #     table.scale(1, 2.5)
        
    #     # --- Apply custom styling ---
    #     cells = table.get_celld()
        
    #     for j in range(len(col_labels)):
    #         cells[(0, j)].set_facecolor('#e0e0e0')
    #         cells[(0, j)].get_text().set_weight('bold')

    #     for i in range(len(plot_df)):
    #         cells[(i + 1, 0)].set_facecolor('#f2f2f2')
    #         cells[(i + 1, 0)].get_text().set_weight('bold')
        
    #     cells[(0, 0)].get_text().set_text(title)
    #     cells[(0, 0)].get_text().set_color('#00008B')
    #     cells[(0, 0)].set_fontsize(14)

    # plt.subplots_adjust(hspace=0.2)
    # output_filename = os.path.join(path_save, 'Calibration_Metrics_Table.pdf')
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    # print(f"Saved calibration metrics table PDF: {output_filename}")


if __name__ == '__main__':
    calibration_data = load_calibration_data()
    if not calibration_data.empty:
        generate_calibration_figure(calibration_data)
        generate_calibration_metrics_table(calibration_data)
        print("\nAll plots and tables generated successfully.")

