# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 17:05:49 2025

@author: mohae
"""

# conformal_analysis_multi.py

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------- Where to save -----------------------
path_save = "./figures_tables"
os.makedirs(path_save, exist_ok=True)

# ============================================================
# ================ Conformal Prediction helpers ==============
# ============================================================

def get_nonconformity_score(df, prob_col, uncertainty_col=None, method_type='prob'):
    """
    Nonconformity score S(x, y):
      - 'prob':   S = 1 - P(true class)
      - 'uq':     S = uncertainty (e.g., uncertainty_class_1)
    """
    y  = df['label'].to_numpy()
    p1 = df[prob_col].to_numpy()
    p0 = 1.0 - p1
    if method_type == 'prob':
        return 1.0 - np.where(y == 1, p1, p0)
    elif method_type == 'uq':
        if uncertainty_col is None or uncertainty_col not in df.columns:
            raise ValueError("Missing uncertainty column for 'uq' method_type.")
        return df[uncertainty_col].to_numpy()
    else:
        raise ValueError("method_type must be 'prob' or 'uq'.")

def calibrated_cp_analysis(df_cal, df_test, prob_col, uncertainty_col, method_type, alpha):
    """
    Split-Conformal:
      - Fit q_hat at (1 - alpha) quantile of calibration nonconformities
      - Test coverage = P(y in prediction set)
      - Mean set size = E[|prediction set|], for binary: in_set_0 + in_set_1
    """
    n_cal = len(df_cal)
    S_cal = get_nonconformity_score(df_cal, prob_col, uncertainty_col, method_type)
    S_cal_sorted = np.sort(S_cal)

    # Conformal quantile (finite-sample)
    # q_hat = S_(k), where k = ceil((n+1)*(1-alpha))
    k_index = int(np.ceil((n_cal + 1) * (1 - alpha)))
    if k_index > n_cal:
        q_hat = S_cal_sorted[-1]
    elif k_index <= 0:
        q_hat = S_cal_sorted[0]
    else:
        q_hat = S_cal_sorted[k_index - 1]

    # Build sets for test points
    if method_type == 'prob':
        p1 = df_test[prob_col].to_numpy()
        s_test_1 = 1.0 - p1
        s_test_0 = 1.0 - (1.0 - p1)
    else:
        s_test = df_test[uncertainty_col].to_numpy()
        s_test_0 = s_test
        s_test_1 = s_test

    in_set_0 = (s_test_0 <= q_hat).astype(int)
    in_set_1 = (s_test_1 <= q_hat).astype(int)

    y = df_test['label'].to_numpy()
    coverage = np.where(y == 0, in_set_0, in_set_1).mean()
    mean_set_size = (in_set_0 + in_set_1).mean()  # binary total size

    return coverage, mean_set_size

def raw_confidence_analysis(df_test, prob_col, confidence_threshold):
    """
    Raw-confidence baseline (no CP):
      - Include class c in set if P(c) >= threshold.
    """
    p1 = df_test[prob_col].to_numpy()
    p0 = 1.0 - p1

    in_set_0 = (p0 >= confidence_threshold).astype(int)
    in_set_1 = (p1 >= confidence_threshold).astype(int)

    y = df_test['label'].to_numpy()
    coverage = np.where(y == 0, in_set_0, in_set_1).mean()
    mean_set_size = (in_set_0 + in_set_1).mean()

    return coverage, mean_set_size

# ============================================================
# =================== One-set compute wrapper =================
# ============================================================

def run_cp_vs_raw_for_set(file_names, nominal_conf_level=0.90,
                          conf_grid=np.linspace(0.80, 1.00, 21)):
    """
    file_names: dict with 4 entries (same keys across all six sets):
        {
          'Prob-CP (None)':       '.../predictions_None.csv',
          'TTA-UQ-CP':            '.../predictions_TTA.csv',
          'MCDropout-UQ-CP':      '.../predictions_McDropout.csv',
          'DeepEnsemble-UQ-CP':   '.../predictions_DeepEnsemble.csv'
        }

    Returns a DataFrame with columns:
        ['method','confidence','coverage','set_size','nominal_conf']
    """
    # Load all requested files
    dfs = {}
    for k, v in file_names.items():
        if not os.path.exists(v):
            print(f"[WARN] Missing file for {k}: {v}")
            continue
        dfs[k] = pd.read_csv(v)

    # Base (Prob-CP) must exist
    if 'Prob-CP (None)' not in dfs:
        raise ValueError("Missing 'Prob-CP (None)' CSV.")

    df_none = dfs['Prob-CP (None)']
    needed_cols = {'label', 'prob_class_1', 'tag_4calibrationSet'}
    if not needed_cols.issubset(df_none.columns):
        raise ValueError("Base file must contain label, prob_class_1, tag_4calibrationSet.")

    # Consolidate working frame
    comp = pd.DataFrame({
        'label': df_none['label'],
        'is_cal': df_none['tag_4calibrationSet'],
        'Prob_Class1_ProbCP': df_none['prob_class_1'],
    })

    # Methods present (optionally skip if file missing)
    if 'TTA-UQ-CP' in dfs:
        comp['Prob_Class1_TTA'] = dfs['TTA-UQ-CP']['prob_class_1']
        if 'uncertainty_class_1' in dfs['TTA-UQ-CP'].columns:
            comp['Uncertainty_TTA'] = dfs['TTA-UQ-CP']['uncertainty_class_1']
    if 'MCDropout-UQ-CP' in dfs:
        comp['Prob_Class1_MCDropout'] = dfs['MCDropout-UQ-CP']['prob_class_1']
        if 'uncertainty_class_1' in dfs['MCDropout-UQ-CP'].columns:
            comp['Uncertainty_MCDropout'] = dfs['MCDropout-UQ-CP']['uncertainty_class_1']
    if 'DeepEnsemble-UQ-CP' in dfs:
        comp['Prob_Class1_DeepEnsemble'] = dfs['DeepEnsemble-UQ-CP']['prob_class_1']
        if 'uncertainty_class_1' in dfs['DeepEnsemble-UQ-CP'].columns:
            comp['Uncertainty_DeepEnsemble'] = dfs['DeepEnsemble-UQ-CP']['uncertainty_class_1']

    df_cal  = comp[comp['is_cal'] == 1].copy()
    df_test = comp[comp['is_cal'] == 0].copy()

    # Methods to analyze
    # (prob methods use prob_class_1 only; uq methods also use uncertainty_class_1)
    methods_to_analyze = []
    if 'Prob_Class1_ProbCP' in comp.columns:
        methods_to_analyze.append(('None', 'Prob_Class1_ProbCP', None, 'prob'))
    if 'Prob_Class1_TTA' in comp.columns:
        methods_to_analyze.append(('TTA', 'Prob_Class1_TTA', 'Uncertainty_TTA', 'uq'))
    if 'Prob_Class1_MCDropout' in comp.columns:
        methods_to_analyze.append(('MCDropout', 'Prob_Class1_MCDropout', 'Uncertainty_MCDropout', 'uq'))
    if 'Prob_Class1_DeepEnsemble' in comp.columns:
        methods_to_analyze.append(('DeepEnsemble', 'Prob_Class1_DeepEnsemble', 'Uncertainty_DeepEnsemble', 'uq'))

    rows = []
    for conf in conf_grid:
        alpha = 1.0 - conf
        for name, pcol, ucol, mtype in methods_to_analyze:
            # CP calibrated
            cov_cp, size_cp = calibrated_cp_analysis(df_cal, df_test, pcol, ucol, mtype, alpha)
            rows.append({'method': f'{name} (CP Calibrated)', 'confidence': conf,
                         'coverage': cov_cp, 'set_size': size_cp})
            # Raw confidence
            cov_raw, size_raw = raw_confidence_analysis(df_test, pcol, conf)
            rows.append({'method': f'{name} (Raw Confidence)', 'confidence': conf,
                         'coverage': cov_raw, 'set_size': size_raw})

    res = pd.DataFrame(rows)
    res['nominal_conf'] = nominal_conf_level
    return res

def report_table(df, fname):
    
    
    # --- 2. Filter for Key Nominal Coverage (0.90) ---
    nominal_conf_level = 0.90
    df_filtered = df[df['confidence'] == nominal_conf_level].copy()
    
    # --- 3. Prepare Data for Pivoting ---
    # Separate the method into base name and calibration type
    df_filtered['base_method'] = df_filtered['method'].apply(lambda x: x.split(' (')[0])
    df_filtered['cal_type'] = df_filtered['method'].apply(lambda x: x.split('(')[-1].replace(')', '').strip())
    
    # Rename columns for clarity in the final table
    df_filtered = df_filtered.rename(columns={'coverage': 'Observed Coverage', 'set_size': 'Mean Set Size'})
    
    # Select only the relevant columns for the paper
    df_summary = df_filtered[['base_method', 'cal_type', 'Observed Coverage', 'Mean Set Size']]
    
    # --- 4. Pivot and Format the Table ---
    
    # Pivot for Observed Coverage: (Method | CP Coverage | Raw Coverage)
    df_coverage = df_summary.pivot(
        index='base_method', 
        columns='cal_type', 
        values='Observed Coverage'
    ).rename(columns={'CP Calibrated': 'Observed Coverage (CP)', 'Raw Confidence': 'Observed Coverage (Raw)'})
    df_coverage.columns.name = None # Remove the column name 'cal_type'
    
    # Pivot for Mean Set Size: (Method | CP Set Size | Raw Set Size)
    df_set_size = df_summary.pivot(
        index='base_method', 
        columns='cal_type', 
        values='Mean Set Size'
    ).rename(columns={'CP Calibrated': 'Mean Set Size (CP)', 'Raw Confidence': 'Mean Set Size (Raw)'})
    df_set_size.columns.name = None
    
    # Combine the two pivoted tables
    df_final = pd.merge(df_coverage, df_set_size, on='base_method').reset_index().rename(columns={'base_method': 'Method'})
    
    # Reorder columns for a logical comparison (CP next to Raw for both metrics)
    df_final = df_final[[
        'Method', 
        'Observed Coverage (CP)', 
        'Observed Coverage (Raw)', 
        'Mean Set Size (CP)', 
        'Mean Set Size (Raw)'
    ]]
    
    # --- 5. Apply Formatting for Presentation ---
    # Format Coverage as percentage (2 decimal places)
    df_final['Observed Coverage (CP)'] = (df_final['Observed Coverage (CP)'] * 100).round(2).astype(str) + '%'
    df_final['Observed Coverage (Raw)'] = (df_final['Observed Coverage (Raw)'] * 100).round(2).astype(str) + '%'
    
    # Format Set Size (3 decimal places)
    df_final['Mean Set Size (CP)'] = df_final['Mean Set Size (CP)'].round(3)
    df_final['Mean Set Size (Raw)'] = df_final['Mean Set Size (Raw)'].round(3)
    
    # Sort by the most important metric: Set Size (CP) - lower is better efficiency
    df_final = df_final.sort_values(by='Mean Set Size (CP)')
    
    # Output the table to CSV for the user
    print(fname[:-4]+'_table10PercentAlpha.csv')
    df_final.to_csv(fname[:-4]+'_table10PercentAlpha.csv', index=False)
    
    print(f"Summary Table for Nominal Coverage 90% (alpha=0.1):")
    print(df_final.to_markdown(index=False))


# ============================================================
# ===================== Plotting (6 panels) ==================
# ============================================================

def plot_six_validity_and_efficiency(results_list, titles,
                                     nominal_conf_level=0.90,
                                     font_cfg=None):
    """
    results_list: list of 6 DataFrames from run_cp_vs_raw_for_set(...)
    titles:       list of 6 strings for each subplot title.
    font_cfg:     dict of mpl rcParams overrides for global font control.
    """
    
    if len(results_list) != 9 or len(titles) != 9:
        raise ValueError("Provide exactly 9 result DataFrames and 9 titles.")

    # --- Global font sizes (scoped to these figures only) ---
    fonts = {
        "font.size": 14,        # base
        "axes.titlesize": 22,   # subplot titles
        "axes.labelsize": 18,   # axis labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
        # Optional:
        # "font.family": "DejaVu Sans",   # or "Times New Roman"
        # "mathtext.fontset": "stix",
        # "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font_cfg:
        fonts.update(font_cfg)

    method_styles = {
        'None':         ('#1f77b4', 'o'),  # blue
        'TTA':          ('#2ca02c', 's'),  # green
        'MCDropout':    ('#d62728', '^'),  # red
        'DeepEnsemble': ('#9467bd', 'D'),  # purple
    }

    print(titles)
    
    
    with mpl.rc_context(rc=fonts):
        # -------- Figure 1: Observed coverage vs nominal --------
        fig1, axes1 = plt.subplots(3, 3, figsize=(22, 13), sharex=True, sharey=True)
        for ax, res, title in zip(axes1.flatten(), results_list, titles):
            conf_grid = np.sort(res['confidence'].unique())

            CSVsaveFilepath = os.path.join(path_save, f'cp_validity_enhancement_{title}.csv')
            res.to_csv(CSVsaveFilepath)
            report_table(res, CSVsaveFilepath )
            
            for method in res['method'].unique():
                sub = res[res['method'] == method]
                base = method.split(' ')[0]
                is_cp = 'CP' in method
                color, marker = method_styles.get(base, ('k', 'o'))

                ax.plot(sub['confidence'], sub['coverage'],
                        marker=marker, markersize=4 if is_cp else 3,
                        linestyle='-' if is_cp else '--',
                        color=color, alpha=1.0 if is_cp else 0.8,
                        label=f'{base} (CP)' if is_cp else f'{base} (Raw)')

            # Ideal line
            ax.plot(conf_grid, conf_grid, 'k--', alpha=0.5, linewidth=2, label='Ideal')

            # Highlight nominal confidence point(s)
            hi = res[np.isclose(res['confidence'], nominal_conf_level)]
            for _, row in hi.iterrows():
                base = row['method'].split(' ')[0]
                color, _ = method_styles.get(base, ('k', 'o'))
                ax.scatter(row['confidence'], row['coverage'], s=140,
                           facecolors='none', edgecolors='k', linewidths=2, zorder=10)
                ax.scatter(row['confidence'], row['coverage'], s=36, color=color, zorder=11)

            ax.set_title(title)
            ax.set_xlabel('Nominal Coverage ($1-\\alpha$)')
            ax.set_ylabel('Observed Coverage')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(min(conf_grid) - 0.01, max(conf_grid) + 0.01)
            ax.set_ylim(0.0, 1.01)

        # Shared legend (bottom)
        handles1, labels1 = axes1.flatten()[0].get_legend_handles_labels()
        fig1.legend(handles1, labels1, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))
        fig1.tight_layout(rect=[0, 0.06, 1, 1])
        fig1.savefig(os.path.join(path_save, 'cp_validity_enhancement_6subplots.pdf'),
                     dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # -------- Figure 2: Mean set size vs nominal --------
        fig2, axes2 = plt.subplots(3, 3, figsize=(22, 14), sharex=True, sharey=True)
        for ax, res, title in zip(axes2.flatten(), results_list, titles):
            conf_grid = np.sort(res['confidence'].unique())

            for method in res['method'].unique():
                sub = res[res['method'] == method]
                base = method.split(' ')[0]
                is_cp = 'CP' in method
                color, marker = method_styles.get(base, ('k', 'o'))

                ax.plot(sub['confidence'], sub['set_size'],
                        marker=marker, markersize=4 if is_cp else 3,
                        linestyle='-' if is_cp else '--',
                        color=color, alpha=1.0 if is_cp else 0.8,
                        label=f'{base} (CP)' if is_cp else f'{base} (Raw)')

            ax.set_title(title)
            ax.set_xlabel('Nominal Coverage ($1-\\alpha$)')
            ax.set_ylabel('Mean Prediction Set Size')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlim(min(conf_grid) - 0.01, max(conf_grid) + 0.01)
            # Typical binary set size ranges within [1, 2]
            #ax.set_ylim(0.9, 2.05)
            ax.set_yticks(np.arange(1.0, 2.1, 0.1))

        handles2, labels2 = axes2.flatten()[0].get_legend_handles_labels()
        fig2.legend(handles2, labels2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
                    fontsize=18)
        fig2.tight_layout(rect=[0, 0.06, 1, 1])
        fig2.savefig(os.path.join(path_save, 'cp_efficiency_comparison_6subplots.pdf'),
                     dpi=300, bbox_inches='tight')
        plt.close(fig2)

# ============================================================
# ============================== Main ========================
# ============================================================

if __name__ == "__main__":
    # Fill these with your real paths. Each dict = one "set" (e.g., disease×model).
    # You need SIX sets. Each set should include up to four keys below.
    sets = [
        ("AMD (ViT)", {
            'Prob-CP (None)':       '../Outputs/CELoss/AMD_ViT_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/CELoss/AMD_ViT_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/CELoss/AMD_ViT_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/CELoss/AMD_ViT_ensemble_1/predictions_DeepEnsemble.csv',
        }),
        ("Diabetes (ViT)", {
            'Prob-CP (None)':       '../Outputs/CELoss/DR_ViT_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/CELoss/DR_ViT_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/CELoss/DR_ViT_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/CELoss/DR_ViT_ensemble_1/predictions_DeepEnsemble.csv',
        }),
        ("Glaucoma (ViT)", {
            'Prob-CP (None)':       '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_DeepEnsemble.csv',
        }),
        ("AMD (RETFoundGreen)", {
            'Prob-CP (None)':       '../Outputs/CELoss/AMD_RETFoundGreen_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/CELoss/AMD_RETFoundGreen_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/CELoss/AMD_RETFoundGreen_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/CELoss/AMD_RETFoundGreen_ensemble_1/predictions_DeepEnsemble.csv',
        }),
        ("Diabetes (RETFoundGreen)", {
            'Prob-CP (None)':       '../Outputs/CELoss/DR_RETFoundGreen_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/CELoss/DR_RETFoundGreen_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/CELoss/DR_RETFoundGreen_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/CELoss/DR_RETFoundGreen_ensemble_1/predictions_DeepEnsemble.csv',
        }),        
        ("Glaucoma (RETFoundGreen)", {
            'Prob-CP (None)':       '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_DeepEnsemble.csv',
        }),
        ("AMD (ViT(FLSD-53))", {
            'Prob-CP (None)':       '../Outputs/FocalLoss/AMD_ViT_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/FocalLoss/AMD_ViT_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/FocalLoss/AMD_ViT_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/FocalLoss/AMD_ViT_ensemble_1/predictions_DeepEnsemble.csv',
        }),
        ("Diabetes (ViT(FLSD-53))", {
            'Prob-CP (None)':       '../Outputs/FocalLoss/DR_ViT_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/FocalLoss/DR_ViT_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/FocalLoss/DR_ViT_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/FocalLoss/DR_ViT_ensemble_1/predictions_DeepEnsemble.csv',
        }),
        ("Glaucoma (ViT(FLSD-53))", {
            'Prob-CP (None)':       '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_None.csv',
            'TTA-UQ-CP':            '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_TTA.csv',
            'MCDropout-UQ-CP':      '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_McDropout.csv',
            'DeepEnsemble-UQ-CP':   '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_DeepEnsemble.csv',
        }),
    ]

    # Run all six sets
    results, titles = [], []
    for title, files in sets:
        results.append(run_cp_vs_raw_for_set(files, nominal_conf_level=0.90))
        titles.append(title)

    # Global font override example (optional):
    # font_cfg = {"axes.titlesize": 24, "legend.fontsize": 18}
    font_cfg = None

    plot_six_validity_and_efficiency(results, titles, nominal_conf_level=0.90, font_cfg=font_cfg)
