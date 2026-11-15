import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

MODELS = ['ViT', 'RETFoundGreen', 'ViT(FLSD-53)']
BASE_OUTPUT_PATH_CELoss = Path('../Outputs/CELoss')
BASE_OUTPUT_PATH_FocalLoss = Path('../Outputs/FocalLoss')

# A detailed map of all methods, their files, and the uncertainty column to use.
# This structure defines the rows and content of the final plot.
METHODS_TO_PLOT = {
    'Variance-Based': {
        'TTA': ('ensemble_1', 'predictions_TTA.csv', 'uncertainty_class_1'),
        'MC Dropout': ('ensemble_1', 'predictions_McDropout.csv', 'uncertainty_class_1'),
        'Deep Ensemble (Var)': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty_class_1'),
    },
    'Entropy-Based': {
        'Predictive Entropy': ('ensemble_1', 'predictions_PredictiveEntropy.csv', 'uncertainty'),
    },
    'Probability-Based': {
        'MSP': ('ensemble_1', 'predictions_MSP.csv', 'uncertainty'),
    },
    'Deep Ensemble Decomposition': {
        'Aleatoric': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty_aleatoric'),
        'Epistemic': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty_epistemic'),
        'Total': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty'),
    },
}


def load_all_data():
    """
    Scans the 'Outputs' directory based on the METHODS_TO_PLOT configuration,
    loads all necessary data, and filters calibration rows.
    """
    all_data = []
    total_removed_count = 0
    print("Starting data loading for uncertainty plots...")

    for plot_group, methods in METHODS_TO_PLOT.items():
        for method_name, (folder_suffix, filename, col_name) in methods.items():
            for disease_code, disease_name in DISEASE_MAP.items():
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
                            
                            if 'tag_4calibrationSet' in df.columns:
                                initial_rows = len(df)
                                df = df[df['tag_4calibrationSet'] != 1].copy()
                                removed_rows = initial_rows - len(df)
                                total_removed_count += removed_rows
                                if removed_rows > 0:
                                    print(f"Filtered {removed_rows} calibration rows from {file_path}")
                            
                            if col_name in df.columns:
                                temp_df = pd.DataFrame({
                                    'Uncertainty': df[col_name],
                                    'Method': method_name,
                                    'Model': model,
                                    'Disease': disease_name,
                                    'Plot_Group': plot_group
                                })
                                all_data.append(temp_df)
                            else:
                                print(f"Column '{col_name}' not found in {file_path}, skipping.")
                        except Exception as e:
                            print(f"Could not process {file_path}. Error: {e}")
                    else:
                        # This is expected for some combinations, so we don't print a warning
                        pass

    if not all_data:
        print("\nERROR: No data was loaded.")
        return pd.DataFrame()

    print(f"\nTotal calibration rows dropped across all files: {total_removed_count}")
    print("Data loading complete.")
    return pd.concat(all_data, ignore_index=True)


import matplotlib as mpl

def plot_uncertainty_violins(all_data):
    print("\nGenerating uncertainty violin plots...")

    # --- Control font sizes globally ---
    mpl.rcParams.update({
        "axes.titlesize": 30,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 27,
    })


    # --- scale epistemic values by 20
    all_data.loc[
        (all_data["Method"] == "Epistemic") &
        (all_data["Plot_Group"] == "Deep Ensemble Decomposition"),
        "Uncertainty"
    ] *= 20

    # Create 4x3 grid
    plot_groups = list(METHODS_TO_PLOT.keys())
    diseases = list(DISEASE_MAP.values())
    fig, axes = plt.subplots(len(plot_groups), len(diseases),
                             figsize=(24, 18), sharey=False)

    palette = sns.color_palette("colorblind", len(MODELS))

    for row_idx, plot_group in enumerate(plot_groups):
        for col_idx, disease in enumerate(diseases):
            ax = axes[row_idx, col_idx]
            subset = all_data[(all_data["Plot_Group"] == plot_group) &
                              (all_data["Disease"] == disease)]
            if subset.empty:
                ax.set_visible(False)
                continue

            # Separate violins for each model
            sns.violinplot(
                data=subset,
                x="Method", y="Uncertainty", hue="Model",
                dodge=True, cut=0, density_norm="width", inner="quartile",
                ax=ax, palette=palette
            )

            if row_idx == 0:
                ax.set_title(disease)
            if col_idx == 0:
                ax.set_ylabel(plot_group)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.grid(True, alpha=0.3)

            # remove subplot legends (keep shared)
            ax.get_legend().remove()
            
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS))

    # --- annotation for epistemic scaling
    fig.text(
        0.5, 0.05,
        "Note: Deep Ensemble Epistemic uncertainties are scaled ×20 for visibility",
        ha="left", style="italic", fontsize=14
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    output_filename = os.path.join(path_save, "uncertainty_violin_plots.pdf")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved violin plot figure: {output_filename}")




if __name__ == '__main__':
    uncertainty_data = load_all_data()
    if not uncertainty_data.empty:
        plot_uncertainty_violins(uncertainty_data)
        print("\nUncertainty plots generated successfully.")

