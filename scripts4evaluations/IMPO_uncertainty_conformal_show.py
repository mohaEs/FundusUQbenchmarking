# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 16:07:06 2025

@author: mohae
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os


path_save = './figures_tables'
os.makedirs(path_save, exist_ok=True)


import matplotlib as mpl

# --- Control font sizes globally ---
mpl.rcParams.update({
    "axes.titlesize": 30,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 27,
})

# --- Configuration ---
DISEASE_MAP = {
    'AMD': 'AMD',
    'DR': 'Diabetes',
    'GL': 'Glaucoma'
}

MODELS = ['ViT', 'RETFoundGreen', 'ViT(FLSD-53)']
BASE_OUTPUT_PATH_CELoss = Path('../Outputs/CELoss')
BASE_OUTPUT_PATH_FocalLoss = Path('../Outputs/FocalLoss')

# A detailed map of all methods, their files, and the uncertainty column to use.
# This structure defines the rows and content of the final plot.
METHODS_TO_PLOT = {
    "Conformal Prediction": {
        "Deterministic": ("ensemble_1", "predictions_None_conformal.csv", "isUncertain"),
        "TTA": ("ensemble_1", "predictions_TTA_conformal.csv", "isUncertain"),
        "MC Dropout": ("ensemble_1", "predictions_McDropout_conformal.csv", "isUncertain"),
        "Deep Ensemble": ("ensemble_1", "predictions_DeepEnsemble_conformal.csv", "isUncertain"),
    }
}


def load_conformal_data():
    """
    Loads conformal prediction results (set_sizes and isUncertain)
    across different methods (Deterministic, TTA, MC Dropout, Deep Ensemble).
    """
    all_data = []
    total_removed_count = 0
    print("Loading conformal prediction data...")
    
    for method_name, (folder_suffix, filename, col_name) in METHODS_TO_PLOT["Conformal Prediction"].items():
        for disease_code, disease_name in DISEASE_MAP.items():
            for model in MODELS:
                if model=='ViT(FLSD-53)':
                    model0 = 'ViT'
                    folder_name = f"{disease_code}_{model0}_{folder_suffix}"
                    file_path = BASE_OUTPUT_PATH_FocalLoss / folder_name / filename
                
                else:
                    folder_name = f"{disease_code}_{model}_{folder_suffix}"
                    file_path = BASE_OUTPUT_PATH_CELoss / folder_name / filename
                #print(file_path)
                
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)

                        if "tag_4calibrationSet" in df.columns:          
                            initial_rows = len(df)
                            df = df[df["tag_4calibrationSet"] != 1].copy()
                            removed_rows = initial_rows - len(df)
                            total_removed_count += removed_rows
                            if removed_rows > 0:
                                print(f"Filtered {removed_rows} calibration rows from {file_path}")
                                
                        #print (col_list)
                        #for col_name in col_list:
                        if col_name in df.columns:
                            temp_df = pd.DataFrame({
                                "Value": df[col_name],
                                "Metric": col_name,
                                "Method": method_name,
                                "Model": model,
                                "Disease": disease_name,
                            })
                            all_data.append(temp_df)
                        else:
                            print(f"Column '{col_name}' not found in {file_path}, skipping.")
                    except Exception as e:
                        print(f"Could not process {file_path}. Error: {e}")

    if not all_data:
        print("No conformal data found.")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)



def create_statistics_summary(df):
    """Create comprehensive statistics for each combination"""
    # Group by Disease, Method, and Model, then count values
    stats = df.groupby(['Disease', 'Method', 'Model', 'Value']).size().unstack(fill_value=0)
    stats = stats.reset_index()
    
    # Ensure we have both columns 1 and 2
    if 1 not in stats.columns:
        stats[1] = 0
    if 2 not in stats.columns:
        stats[2] = 0
    
    # Rename columns for clarity
    stats = stats.rename(columns={1: 'Value_1', 2: 'Value_2'})
    stats['Total'] = stats['Value_1'] + stats['Value_2']
    stats['Percentage_1'] = (stats['Value_1'] / stats['Total'] * 100).round(1)
    stats['Percentage_2'] = (stats['Value_2'] / stats['Total'] * 100).round(1)
    
    return stats


def print_detailed_statistics(df):
    """Print detailed statistics table"""
    stats = create_statistics_summary(df)
    
    print("\n" + "="*80)
    print("DETAILED STATISTICS TABLE")
    print("="*80)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(stats.to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"Total records (values 1 and 2): {stats['Total'].sum()}")
    print(f"Total Value 1 records: {stats['Value_1'].sum()}")
    print(f"Total Value 2 records: {stats['Value_2'].sum()}")
    print(f"Overall percentage Value 1: {stats['Value_1'].sum() / stats['Total'].sum() * 100:.1f}%")
    print(f"Overall percentage Value 2: {stats['Value_2'].sum() / stats['Total'].sum() * 100:.1f}%")

# Main execution
if __name__ == "__main__":
    
    # Load and process data
    df_conformal_data = load_conformal_data()
    # print_detailed_statistics(df_conformal_data)
    # stats=create_statistics_summary(df_conformal_data)
    

    
    # Explicitly take all 3 Diseases × 4 Methods = 12 panels
    groups = [(d, m) for d in df_conformal_data["Disease"].unique()
                       for m in df_conformal_data["Method"].unique()]
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), sharey=True)
    
    for ax, (disease, method) in zip(axes.flatten(), groups):
        subset = df_conformal_data[(df_conformal_data["Disease"] == disease) &
                                   (df_conformal_data["Method"] == method)]
    
        # Compute proportions manually
        prop_df = (
            subset.groupby("Model")["Value"]
            .value_counts(normalize=True)
            .rename("Proportion")
            .reset_index()
        )
    
        sns.barplot(
            data=prop_df,
            x="Value", y="Proportion", hue="Model",
            palette="colorblind",
            ax=ax
        )
    
        # Titles and labels
        ax.set_title(f"{disease} ({method})")
        ax.set_xlabel("Uncertain")
        ax.set_ylabel("Proportion")
    
        # Replace 0/1 with No/Yes
        ax.set_xticklabels(["No", "Yes"])
    
        ax.set_ylim(0, 1.1)  # extra space for labels
        ax.grid(axis="y", alpha=0.3)
    
        # Remove subplot legends (keep global)
        ax.get_legend().remove()
    
        # --- Annotate each bar with value ---
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height) and height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.,  # center of bar
                    height + 0.02,                   # a little above the bar
                    f"{height:.2f}",                 # format as 2 decimals
                    ha="center", va="bottom", fontsize=10
                )
    
    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(df_conformal_data['Model'].unique()))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_filename = os.path.join(path_save, "uncertainty_conformals.pdf")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved violin plot figure: {output_filename}")
    #plt.show()

    



