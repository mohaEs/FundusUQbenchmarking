import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

path_save = './figures_tables/MEE'
os.makedirs(path_save, exist_ok=True)

# --- Configuration ---
DISEASE_MAP = {
    'GL': 'Glaucoma'
}

# Define the methods and corresponding files we want to plot
METHODS_TO_PLOT = {
    'Baseline': ('ensemble_MEE', 'predictions_None_conformal.csv'),
    'MC Dropout': ('ensemble_MEE', 'predictions_McDropout_conformal.csv'),
    'TTA': ('ensemble_MEE', 'predictions_TTA_conformal.csv'),
    'Deep Ensemble': ('ensemble_MEE', 'predictions_DeepEnsemble_conformal.csv')
}

MODELS = ['ViT', 'RETFoundGreen', 'ViT(FLSD-53)']
BASE_OUTPUT_PATH_CELoss = Path('../Outputs/CELoss')
BASE_OUTPUT_PATH_FocalLoss = Path('../Outputs/FocalLoss')


def load_performance_data():
    """
    Scans the 'Outputs' directory to load data for AUROC/AUPRC calculations,
    filtering out rows where 'tag_4calibrationSet' is 1, and reports the count.
    """
    all_data = []
    total_removed_count = 0
    print("Starting data loading for performance curves...")

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
                        
                        df = df[df['isUncertain'] != 1].copy()
                        
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
import matplotlib as mpl

def generate_performance_figure(data_df):
    """
    Generates and saves a single 2x1 figure containing AUROC and AUPRC subplots
    (stacked vertically) for the one disease available (e.g., Glaucoma),
    with a shared legend at the bottom.
    """
    print("\nGenerating performance curve figure...")

    # --- set font sizes ---
    mpl.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    # --- 1 disease → 2 rows, 1 column (ROC top, PRC bottom)
    fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=False, sharey=False)

    method_colors = sns.color_palette("colorblind", len(METHODS_TO_PLOT))
    model_linestyles = {'ViT': '-', 'RETFoundGreen': '--', 'ViT(FLSD-53)': ':' }
    color_map = {method: color for method, color in zip(METHODS_TO_PLOT.keys(), method_colors)}

    disease_name = list(DISEASE_MAP.values())[0]
    disease_data = data_df[data_df['Disease'] == disease_name]

    # --- ROC subplot ---
    ax_roc = axes[0]
    for method in METHODS_TO_PLOT.keys():
        for model in MODELS:
            subset = disease_data[(disease_data['Method'] == method) & (disease_data['Model'] == model)]
            if not subset.empty:
                y_true = subset['label']
                y_score = subset['prob_class_1']
                fpr, tpr, _ = roc_curve(y_true, y_score)
                ax_roc.plot(fpr, tpr, color=color_map[method], linestyle=model_linestyles[model],
                            label=f'{model} - {method}', lw=2)
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curve - {disease_name}')
    ax_roc.grid(True, alpha=0.3)

    # --- PRC subplot ---
    ax_prc = axes[1]
    for method in METHODS_TO_PLOT.keys():
        for model in MODELS:
            subset = disease_data[(disease_data['Method'] == method) & (disease_data['Model'] == model)]
            if not subset.empty:
                y_true = subset['label']
                y_score = subset['prob_class_1']
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                ax_prc.plot(recall, precision, color=color_map[method], linestyle=model_linestyles[model],
                            label=f'{model} - {method}', lw=2)
    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title(f'PR Curve - {disease_name}')
    ax_prc.grid(True, alpha=0.3)

    # --- shared legend at bottom ---
    handles, labels = ax_roc.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_filename = os.path.join(path_save, f'AUROC_AUPRC_{disease_name}_MEE.pdf')
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved performance figure: {output_filename}")



def generate_performance_tables(data_df):
    """
    Calculates AUROC and AUPRC scores, exports them as a single PNG image with
    a multi-level header, and saves the raw data to CSV files.
    """
    print("\nGenerating performance tables...")
    
    results = []
    # Calculate scores for each combination
    for disease in data_df['Disease'].unique():
        for model in data_df['Model'].unique():
            for method in data_df['Method'].unique():
                subset = data_df[(data_df['Disease'] == disease) & 
                                 (data_df['Model'] == model) & 
                                 (data_df['Method'] == method)]
                if not subset.empty:
                    y_true = subset['label']
                    y_score = subset['prob_class_1']
                    
                    roc_auc = auc(*roc_curve(y_true, y_score)[:2])
                    prc_auc = average_precision_score(y_true, y_score)
                    
                    results.append({
                        'Disease': disease, 'Model': model, 'Method': method,
                        'AUROC': roc_auc, 'AUPRC': prc_auc
                    })

    results_df = pd.DataFrame(results)

    # --- Create pivot tables with the original "models as columns" structure ---
    auroc_pivot = results_df.pivot_table(index='Disease', columns=['Model', 'Method'], values='AUROC')
    auprc_pivot = results_df.pivot_table(index='Disease', columns=['Model', 'Method'], values='AUPRC')
    
    # --- Define consistent ordering for index and columns ---
    column_order = [(model, method) for model in MODELS for method in METHODS_TO_PLOT.keys()]
    row_order = [DISEASE_MAP[code] for code in ['GL']]#'AMD', 'DR',
    
    auroc_pivot = auroc_pivot.reindex(index=row_order, columns=column_order)
    auprc_pivot = auprc_pivot.reindex(index=row_order, columns=column_order)

    # --- Save tables to CSV files ---
    auroc_csv_filename = os.path.join(path_save, 'AUROC_performance_MEE.csv')
    auprc_csv_filename = os.path.join(path_save, 'AUPRC_performance_MEE.csv')
    auroc_pivot.to_csv(auroc_csv_filename)
    auprc_pivot.to_csv(auprc_csv_filename)
    print(f"Saved AUROC data to: {auroc_csv_filename}")
    print(f"Saved AUPRC data to: {auprc_csv_filename}")

    # # --- Plotting the tables as a PNG ---
    # fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    # fig.patch.set_facecolor('white')
    
    # # Set both title colors to dark blue
    # title_colors = ['#00008B', '#00008B']
    
    # for ax, pivot_table, title, color in zip(axes, [auroc_pivot, auprc_pivot], ['AUROC', 'AUPRC'], title_colors):
    #     ax.axis('off')
        
    #     # --- Prepare data for a stable table structure by resetting the index ---
    #     plot_df = pivot_table.round(3).reset_index()
        
    #     # Create wrapped column headers, adding a placeholder for the new first column
    #     col_labels = [''] + [f'{model}\n{method}' for model, method in pivot_table.columns]
        
    #     # Define column widths: narrower first column, wider RETFoundGreen columns
    #     col_widths = [0.09] + [0.125] * 8

    #     # Create the main table without using the problematic `rowLabels` argument
    #     table = ax.table(cellText=plot_df.values,
    #                      colLabels=col_labels,
    #                      loc='center',
    #                      cellLoc='center',
    #                      colWidths=col_widths)
        
    #     table.auto_set_font_size(False)
    #     table.set_fontsize(11)
    #     table.scale(1, 2.5) # Scale for taller header row
        
    #     # --- Apply custom styling using the internal cell dictionary ---
    #     cells = table.get_celld()
        
    #     # Style the entire header row (row 0)
    #     for j in range(len(col_labels)):
    #         cells[(0, j)].set_facecolor('#e0e0e0')
    #         cells[(0, j)].get_text().set_weight('bold')

    #     # Style the first column (the disease names) to look like row headers
    #     for i in range(len(plot_df)):
    #         cells[(i + 1, 0)].set_facecolor('#f2f2f2')
    #         cells[(i + 1, 0)].get_text().set_weight('bold')
        
    #     # Style the top-left cell (0,0) with the title
    #     cells[(0, 0)].get_text().set_text(title)
    #     cells[(0, 0)].get_text().set_color(color)
    #     cells[(0, 0)].set_fontsize(14)
            

    # plt.subplots_adjust(hspace=0.1)
    # output_filename = os.path.join(path_save, 'AUROC_AUPRC_Performance_Tables.pdf')
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    # print(f"Saved formatted performance tables PDF: {output_filename}")


if __name__ == '__main__':
    performance_data = load_performance_data()

    if not performance_data.empty:
        generate_performance_figure(performance_data)
        generate_performance_tables(performance_data)
        print("\nAll plots and tables have been generated successfully.")

