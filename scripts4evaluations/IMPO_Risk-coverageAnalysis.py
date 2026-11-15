import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import auc
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

METHODS_TO_PLOT = {
    'Baseline (1-prob)': ('ensemble_1', 'predictions_None.csv', 'prob_class_1'),
    'MSP': ('ensemble_1', 'predictions_MSP.csv', 'uncertainty'),
    'Predictive Entropy': ('ensemble_1', 'predictions_PredictiveEntropy.csv', 'uncertainty'),
    'MC Dropout': ('ensemble_1', 'predictions_McDropout.csv', 'uncertainty_class_1'),
    'TTA': ('ensemble_1', 'predictions_TTA.csv', 'uncertainty_class_1'),
    'Deep Ensemble (var)': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty_class_1'),
    'Deep Ensemble (aleatoric)': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty_aleatoric'),
    'Deep Ensemble (Epistemic)': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty_epistemic'),
    'Deep Ensemble (Total)': ('ensemble_1', 'predictions_DeepEnsemble.csv', 'uncertainty'),    
    #'Deep Ensemble (Total) & Conformal Inference': ('ensemble_1', 'predictions_DeepEnsemble_conformal.csv', 'isUncertain'),
}



MODELS = ['ViT', 'RETFoundGreen', 'ViT(FLSD-53)']
BASE_OUTPUT_PATH_CELoss = Path('../Outputs/CELoss')
BASE_OUTPUT_PATH_FocalLoss = Path('../Outputs/FocalLoss')

import numpy as np
from matplotlib.ticker import FuncFormatter

def apply_probability_scale(ax, mode="power", alpha=0.5):
    """
    Remap the y-axis to emphasize small risks.
    mode="power" uses y^alpha (alpha<1 expands near 0, shrinks near 1).
    mode="logit" expands both ends (needs clipping away from 0 and 1).
    """
    if mode == "power":
        fwd  = lambda y: np.power(y, alpha)
        inv  = lambda y: np.power(y, 1.0/alpha)
        ax.set_yscale("function", functions=(fwd, inv))
        # nice tick positions in ORIGINAL risk space:
        ticks = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))
    elif mode == "logit":
        # expands near both 0 and 1; use only if you want to see behavior very close to 1 as well
        ax.set_yscale("logit")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))



def load_risk_coverage_data():
    """
    Scans 'Outputs' to load data for risk-coverage analysis.
    For conformal files (binary flag present), sets:
      - uncertainty = isUncertain (0/1)
      - secondary_score = uncertainty_class_1 (if present)
    """
    all_data = []
    print("Starting data loading for risk-coverage analysis...")

    flag_cols = ["isUncertain", "is_uncertain", "isUncertaint", "conformal_flag"]

    for disease_code, disease_name in DISEASE_MAP.items():
        for method_name, (folder_suffix, filename, uncertainty_col) in METHODS_TO_PLOT.items():
            for model in MODELS:
                if model=='ViT(FLSD-53)':
                    model0 = 'ViT'
                    folder_name = f"{disease_code}_{model0}_{folder_suffix}"
                    file_path = BASE_OUTPUT_PATH_FocalLoss / folder_name / filename
                
                else:
                    folder_name = f"{disease_code}_{model}_{folder_suffix}"
                    file_path = BASE_OUTPUT_PATH_CELoss / folder_name / filename

                if not file_path.exists():
                    # optional: keep quiet if many combos are absent
                    continue

                try:
                    df = pd.read_csv(file_path)

                    # drop calibration rows if tagged
                    if "tag_4calibrationSet" in df.columns:
                        df = df[df["tag_4calibrationSet"] != 1].copy()

                    # basic required columns for all methods
                    if not all(c in df.columns for c in ["label", "prediction"]):
                        print(f"Missing label/prediction in {file_path}, skipping.")
                        continue

                    # # --- CONFORMAL: binary flag present
                    # present_flags = [c for c in flag_cols if c in df.columns]
                    # if present_flags:
                    #     flag = df[present_flags[0]].astype(int).values
                    #     temp = {
                    #         "label": df["label"].values,
                    #         "prediction": df["prediction"].values,
                    #         "uncertainty": flag,             # 0/1 flag
                    #         "Method": method_name,
                    #         "Model": model,
                    #         "Disease": disease_name,
                    #     }
                    #     # optional deterministic ranking within groups
                    #     if "uncertainty_class_1" in df.columns:
                    #         temp["secondary_score"] = df["uncertainty_class_1"].astype(float).values
                    #     temp_df = pd.DataFrame(temp)
                    #     all_data.append(temp_df)
                    #     continue  # done with this file

                    # --- NON-CONFORMAL: use the mapped uncertainty column
                    
                        
                    if uncertainty_col in df.columns:
                        temp_df = pd.DataFrame({
                            "label": df["label"].values,
                            "prediction": df["prediction"].values,
                            "uncertainty": df[uncertainty_col].astype(float).values,
                            "Method": method_name,
                            "Model": model,
                            "Disease": disease_name
                        })
                        if uncertainty_col == 'prob_class_1':
                            temp_df["uncertainty"]=1-df[uncertainty_col].astype(float).values
                        
                        all_data.append(temp_df)
                    else:
                        print(f"Required column '{uncertainty_col}' not found in {file_path}, skipping.")
                        
                    
                        

                except Exception as e:
                    print(f"Could not process {file_path}. Error: {e}")

    if not all_data:
        print("\nERROR: No data was loaded.")
        return pd.DataFrame()

    print("Data loading complete.")
    return pd.concat(all_data, ignore_index=True)



def calculate_risk_coverage_points(df):
    """Calculates risk vs coverage points for plotting."""
    df_sorted = df.sort_values(by='uncertainty', ascending=False)
    n_total = len(df_sorted)
    
    coverages = []
    risks = []
    
    for i in range(1, n_total + 1):
        subset = df_sorted.iloc[n_total-i:]
        coverage = len(subset) / n_total
        
        if len(subset) == 0:
            continue
            
        correct_predictions = (subset['prediction'] == subset['label']).sum()
        accuracy = correct_predictions / len(subset)
        risk = 1 - accuracy
        
        coverages.append(coverage)
        risks.append(risk)
        
    return np.array(coverages), np.array(risks)





def calculate_selective_prediction_metrics(coverages, risks):
    """Calculates AURC, Risk at 90% Coverage, and Coverage at 5% Risk."""
    sorted_indices = np.argsort(coverages)
    coverages = coverages[sorted_indices]
    risks = risks[sorted_indices]

    aurc = auc(1 - coverages, risks) if len(coverages) > 1 else 0
    risk_at_90_cov = np.interp(0.9, coverages, risks, left=1.0, right=risks[-1] if len(risks)>0 else 1.0)
    low_risk_indices = np.where(risks <= 0.05)[0]
    cov_at_5_risk = coverages[low_risk_indices[-1]] if len(low_risk_indices) > 0 else 0.0
    
    return aurc, risk_at_90_cov, cov_at_5_risk


import matplotlib as mpl

def generate_risk_coverage_figure(
    data_df,
    font_cfg=None,
):
    """Generates and saves a 2x3 figure of risk-coverage curves for methods."""
    print("\nGenerating risk-coverage figure...")

    # --- Global font sizes (scoped to this figure only) ---
    default_fonts = {
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 20,
    }
    if font_cfg:
        default_fonts.update(font_cfg)

    with mpl.rc_context(rc=default_fonts):
        fig, axes = plt.subplots(3, 3, figsize=(24, 15), sharex=True, sharey=True)

        # consistent colors by method
        color_map = {
            method: color
            for method, color in zip(
                METHODS_TO_PLOT.keys(),
                sns.color_palette("colorblind", len(METHODS_TO_PLOT))
            )
        }

        for row_idx, model in enumerate(MODELS):
            for col_idx, disease_name in enumerate(DISEASE_MAP.values()):
                ax = axes[row_idx, col_idx]

                for method in METHODS_TO_PLOT.keys():
                    subset = data_df[
                        (data_df["Model"] == model)
                        & (data_df["Disease"] == disease_name)
                        & (data_df["Method"] == method)
                    ]
                    if subset.empty:
                        continue

                    coverages, risks = calculate_risk_coverage_points(subset)
                    ax.plot(coverages, risks, color=color_map[method], label=method, lw=2.5)

                # Titles/labels inherit sizes from rcParams
                ax.set_title(f"{model} - {disease_name}")
                ax.set_xlabel("Coverage")
                ax.set_ylabel("Risk (Error Rate)")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                # Emphasize small risks (your helper)
                apply_probability_scale(ax, mode="power", alpha=0.25)

        # --- Single, shared legend ---
        handles, labels = [], []
        for ax in axes.flatten():
            h, l = ax.get_legend_handles_labels()
            for i, lab in enumerate(l):
                if lab not in labels:
                    labels.append(lab)
                    handles.append(h[i])

        if handles:
            fig.legend(
                handles, labels,
                loc="lower center",
                ncol=5,               # tweak as needed
                bbox_to_anchor=(0.5, -0.02)  # just below plots
            )

        # Leave room for legend; bbox_inches ensures nothing gets cropped
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        output_filename = os.path.join(path_save, "selective_prediction_Curves.pdf")
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved risk-coverage figure: {output_filename}")


def generate_selective_prediction_table(data_df):
    """
    Calculates metrics for  methods and saves results to a PNG and CSV.
    """
    print("\nGenerating  selective prediction metrics table...")
    
    results = []
    for disease in data_df['Disease'].unique():
        for model in data_df['Model'].unique():
            for method in data_df['Method'].unique():
                subset = data_df[(data_df['Disease'] == disease) & 
                                 (data_df['Model'] == model) & 
                                 (data_df['Method'] == method)]
                if not subset.empty:
                    coverages, risks = calculate_risk_coverage_points(subset)
                    aurc, risk_at_90, cov_at_5 = calculate_selective_prediction_metrics(coverages, risks)
                    results.append({
                        'Disease': disease, 'Model': model, 'Method': method,
                        'AURC': aurc, 'Risk @ 90% Cov': risk_at_90, 'Cov @ 5% Risk': cov_at_5
                    })

    results_df = pd.DataFrame(results)
    
    # --- Save comprehensive table to CSV file ---
    melted_df = results_df.melt(id_vars=['Disease', 'Model', 'Method'], 
                                value_vars=['AURC', 'Risk @ 90% Cov', 'Cov @ 5% Risk'],
                                var_name='Metric', value_name='Score')
    
    full_pivot = melted_df.pivot_table(index=['Metric', 'Disease'], columns=['Model', 'Method'], values='Score')
    csv_filename = os.path.join(path_save,'selective_prediction_metrics.csv')
    full_pivot.to_csv(csv_filename)
    print(f"Saved selective prediction metrics to: {csv_filename}")

    # # --- Plotting the tables ---
    # fig, axes = plt.subplots(3, 2, figsize=(12, 9)) # Narrower figure for fewer columns
    # fig.patch.set_facecolor('white')
    
    # metrics_to_plot_map = {
    #     'AURC': 'AURC',
    #     'Risk @ 90% Cov': 'Risk @ 90%\nCoverage',
    #     'Cov @ 5% Risk': 'Coverage @\n5% Risk'
    # }
    # method_wrap_map = {
    #     'Predictive Entropy': 'Predictive\nEntropy ',
    #     'MSP': 'MSP'
    # }
    
    # metrics_to_plot = ['AURC', 'Risk @ 90% Cov', 'Cov @ 5% Risk']
    # colors = ['#00008B', '#8B0000', '#006400']

    # for row_idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    #     for col_idx, model in enumerate(MODELS):
    #         ax = axes[row_idx, col_idx]
    #         ax.axis('off')
            
    #         model_metric_df = results_df[results_df['Model'] == model]
    #         pivot = model_metric_df.pivot_table(index='Disease', columns='Method', values=metric)
            
    #         row_order = [DISEASE_MAP[code] for code in ['AMD', 'DR', 'GL']]
    #         col_order = list(METHODS_TO_PLOT.keys())
    #         pivot = pivot.reindex(index=row_order, columns=col_order)
            
    #         plot_df = pivot.round(3).reset_index()
            
    #         wrapped_col_labels = [''] + [method_wrap_map.get(c, c) for c in pivot.columns]
            
    #         table = ax.table(cellText=plot_df.values,
    #                          colLabels=wrapped_col_labels,
    #                          loc='center',
    #                          cellLoc='center')
            
    #         table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 2.8)
            
    #         cells = table.get_celld()
    #         cells[(0, 0)].get_text().set_text(metrics_to_plot_map[metric])
    #         cells[(0, 0)].get_text().set_color(color)
    #         cells[(0, 0)].set_fontsize(14)
    #         for j in range(len(wrapped_col_labels)):
    #             cells[(0, j)].set_facecolor('#e0e0e0'); cells[(0, j)].get_text().set_weight('bold')

    #         for i in range(len(plot_df)):
    #             cells[(i + 1, 0)].set_facecolor('#f2f2f2'); cells[(i + 1, 0)].get_text().set_weight('bold')
            
    #         if row_idx == 0:
    #             ax.set_title(model, fontsize=18, weight='bold')

    # plt.subplots_adjust(wspace=0.1, hspace=0.3)
    # output_filename = 'Selective_Prediction_Metrics_Table.pdf'
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    # print(f"Saved selective prediction metrics table PDF: {output_filename}")


if __name__ == '__main__':
    risk_data = load_risk_coverage_data()
    if not risk_data.empty:
        generate_risk_coverage_figure(risk_data)
        generate_selective_prediction_table(risk_data)
        print("\nRisk-coverage analysis for methods complete.")

