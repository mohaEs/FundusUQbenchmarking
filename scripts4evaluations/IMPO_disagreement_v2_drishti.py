# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:03:19 2025

@author: mohae
"""
import pandas as pd

#####################################
#################### DRISTHI 
#####################################

col_names = ["ind", "filename", "id", "label_1","label_2",
             "label_3","label_4","label_5",'label']  # your chosen names
df_drishti = pd.read_csv("Drishti-GS1_diagnosis.csv", header=None, names=col_names)
df_drishti=df_drishti[5:]

label_cols = [f"label_{i}" for i in range(1, 6)]
df_drishti[label_cols] = df_drishti[label_cols].astype(int)
# Count how many +1 and -1 per row
df_drishti["count_pos"] = df_drishti[label_cols].eq(1).sum(axis=1)
df_drishti["count_neg"] = df_drishti[label_cols].eq(-1).sum(axis=1)

# Tag rows where it's 3 vs 2
df_drishti["disagree_3vs2"] = ((df_drishti["count_pos"] == 3) & (df_drishti["count_neg"] == 2)) | \
                      ((df_drishti["count_pos"] == 2) & (df_drishti["count_neg"] == 3))

df_drishti["disagree_4vs1"] = ((df_drishti["count_pos"] == 4) & (df_drishti["count_neg"] == 1)) | \
                      ((df_drishti["count_pos"] == 1) & (df_drishti["count_neg"] == 4))                      
                      
num_disagreements = df_drishti["disagree_3vs2"].sum()
print("Number of 3vs2 disagreements:", num_disagreements)

num_disagreements = df_drishti["disagree_4vs1"].sum()
print("Number of 4vs1 disagreements:", num_disagreements)


df_drishti['filename'] = df_drishti['filename']  
import numpy as np
df_drishti['any_disagremment'] = np.logical_or(df_drishti['disagree_3vs2'], df_drishti['disagree_4vs1'])

num_disagreements = df_drishti["any_disagremment"].sum()
print("Number of any disagreements:", num_disagreements)

#####################################
#################### Analysis for Drishti
#####################################
#####################################
#################### 
####################

#saveFilename = './figures_tables/ppv_GL_ViT.pdf'
from scipy.stats import ttest_ind
import os 
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np

import io
from PIL import Image


col_interest =  'any_disagremment' # 'disagree_3vs2' #

# saveFilename = './figures_tables/agreement_GL_ViT_dristhti_levelAny.pdf'
# csv_files= [    
# #'../Outputs/GL_ViT_ensemble_1/predictions_None_conformal.csv',
#  '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_McDropout_conformal.csv',
#  '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_ViT_ensemble_1/predictions_TTA_conformal.csv',
# ]

# saveFilename = './figures_tables/agreement_GL_RETFoundGreen_dristhti_levelAny.pdf'
# csv_files= [    
# #'../Outputs/GL_ViT_ensemble_1/predictions_None_conformal.csv',
#  '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_McDropout_conformal.csv',
#  '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
#  '../Outputs/CELoss/GL_RETFoundGreen_ensemble_1/predictions_TTA_conformal.csv',
# ]

saveFilename = './figures_tables/agreement_GL_ViT(FLSD-53)_dristhti_levelAny.pdf'
csv_files= [    
#'../Outputs/GL_ViT_ensemble_1/predictions_None_conformal.csv',
 '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_McDropout_conformal.csv',
 '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
 '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
 '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
 '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_DeepEnsemble_conformal.csv',    
 '../Outputs/FocalLoss/GL_ViT_ensemble_1/predictions_TTA_conformal.csv',
]


Models =[ 'MC Dropout', 'Deep Ensemble (var)', 'Deep Ensemble (aleatoric)',
         'Deep Ensemble (epistemic)',
         'Deep Ensemble (total)', 'TTA']#'Baseline (no UQ)',

# --- make one big canvas: 2 rows (box+bar), N columns (models)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

N = len(csv_files)

# Make a compact 2 x N grid; share y within each row to save space
fig, axes = plt.subplots(
    2, N,
    figsize=(3.0 * N, 6),           # ↓ narrower per column
    sharey=False,#'row',
    gridspec_kw={'wspace': 0.15}    # ↓ tighten horizontal gaps
)

for i, csv_file in enumerate(csv_files):
    model_name = Models[i]

    df = pd.read_csv(csv_file)
    if model_name == 'Deep Ensemble (total)':
        df['uncertainty_class_1'] = df['uncertainty']
    if model_name == 'Deep Ensemble (aleatoric)':
        df['uncertainty_class_1'] = df['uncertainty_aleatoric']        
    if model_name == 'Deep Ensemble (epistemic)':
        df['uncertainty_class_1'] = df['uncertainty_epistemic']

    # merge with df_drishti (reuse your merge code here)
    df_calibrationSet = df[df['tag_4calibrationSet'] == 1].copy()
    df = df[df['tag_4calibrationSet'] != 1].copy()
    
    df_resutlt_dristhti = df[df.dataset=='Drishti']
    
    #df_resutlt_dristhti["filename"] = df_resutlt_dristhti["image_path_orig"].apply(os.path.basename)
    df_resutlt_dristhti["filename_noext"] = df_resutlt_dristhti["image_path_orig"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df_resutlt_dristhti["filename_noext"] = df_resutlt_dristhti["filename_noext"].astype(str) + "'"
    
    df_merged = pd.merge(
    df_resutlt_dristhti, df_drishti,
    left_on="filename_noext",    # column in A
    right_on="filename",  # column in B
    how="inner"   # or "left"/"right"/"outer" depending on your need
    )
    df_merged["disagree_3vs2"] = df_merged["disagree_3vs2"].map({True: 1, False: 0})
    df_merged["disagree_4vs1"] = df_merged["disagree_4vs1"].map({True: 1, False: 0})    
    df_merged["any_disagremment"] = df_merged["any_disagremment"].map({True: 1, False: 0})
    print(df_merged["any_disagremment"].sum())
    print(df_merged["disagree_3vs2"].sum())
    print(df_merged["disagree_4vs1"].sum())
    ####################
    
    
    # physician disagreement    
    unc_disagree = df_merged.loc[df_merged[col_interest]==1, "uncertainty_class_1"]
    unc_agree    = df_merged.loc[df_merged[col_interest]==0, "uncertainty_class_1"]
    _, pval_t = ttest_ind(unc_disagree, unc_agree, equal_var=False)
    data_conti = [unc_agree, unc_disagree]
    groups = ["No", "Yes"]

    # conformal set size
    df_merged["conf_set_size"] = df_merged["conf_pred_set"].apply(lambda x: len(eval(x)))
    conf_disagree = df_merged.loc[df_merged[col_interest]==1, "conf_set_size"]
    conf_agree    = df_merged.loc[df_merged[col_interest]==0, "conf_set_size"]
    _, pval_u = mannwhitneyu(conf_disagree, conf_agree, alternative="two-sided")

    df_merged["conf_size_bin"] = (df_merged["conf_set_size"] > 1).astype(int)
    contingency = pd.crosstab(df_merged[col_interest], df_merged["conf_size_bin"])
    chi2, pval_chi, _, _ = chi2_contingency(contingency)

    ratios = []
    for g in [0, 1]:  
        subset = df_merged.loc[df_merged[col_interest]==g, "conf_set_size"]
        total = len(subset)
        size1 = (subset == 1).sum() / total
        size2 = (subset == 2).sum() / total
        ratios.append([size1, size2])
    ratios = np.array(ratios)

    ###################
    #  Pltoting
    ######################
    
    x = np.arange(len(groups))
    width = 0.2
    
    ax_box = axes[0, i]
    ax_bar = axes[1, i]
    
    # ---- Subplot 1: boxplot (continuous UQ) ----
    parts = ax_box.violinplot(data_conti, showmeans=False, showmedians=True)
    
    # set x-ticks to your group labels
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(groups)
    
    if i == 0:
        ax_box.set_ylabel("Uncertainty Score \n ", fontsize=13)
    else:
        ax_box.set_ylabel(None)
        ax_box.tick_params(labelleft=False)
    
#    ax_box.set_title(f"{model_name}\n t-Test p={pval_t:.3g}", fontsize=13)
    
    ax_box.set_title(f"{model_name}", fontsize=12, pad=2)   # Top line, bigger font
    ax_box.annotate(f"t-Test p={pval_t:.3g}",
                xy=(0.5, .8),
                xycoords="axes fraction",
                ha="center",
                fontsize=10)   # Second line, smaller font

    
    ax_box.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax_box.tick_params(labelleft=False)  # hides tick labels only
    
    # ---- Subplot 2: stacked bar (conformal set sizes) ----
    p1 = ax_bar.bar(x, ratios[:, 0], width, label="Set size = 1",
                    color="tab:blue", alpha=0.7)
    p2 = ax_bar.bar(x, ratios[:, 1], width, bottom=ratios[:, 0],
                    label="Set size = 2", color="tab:orange", alpha=0.7)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['No', 'Yes'])
    if i == 0:
        ax_bar.set_ylabel("Conformal Prediction Set Size \n Proportional \n", fontsize=13)
    else:
        ax_bar.set_ylabel(None)
        ax_bar.tick_params(labelleft=False)
    ax_bar.set_xlabel("Having Physician \n Disagreement", fontsize=12)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title(f"Chi² p={pval_chi:.3g}, MW p={pval_u:.3g}", fontsize=10,
                     pad=0  )
    
    
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax_bar.tick_params(labelleft=False)  # hides tick labels only

# ---- Figure-level legend (use explicit handles so colors are correct) ----
legend_handles = [
    Patch(facecolor='tab:blue', alpha=0.7, label='Set size = 1'),
    Patch(facecolor='tab:orange', alpha=0.7, label='Set size = 2')
]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True,
           fontsize=12)

# Tighten layout to reserve space for legend
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.subplots_adjust(hspace=0.3)  # try values 0.3–0.6

# ---- Draw a single box around each model's column (both rows together) ----
# Do this AFTER tight_layout so positions are final
# fig.canvas.draw()
# for i in range(N):
#     top_ax = axes[0, i]
#     bot_ax = axes[1, i]
#     pos_top = top_ax.get_position()
#     pos_bot = bot_ax.get_position()

#     x0 = min(pos_top.x0, pos_bot.x0)
#     x1 = max(pos_top.x1, pos_bot.x1)
#     y0 = min(pos_top.y0, pos_bot.y0)
#     y1 = max(pos_top.y1, pos_bot.y1)

#     # Small padding around the column box
#     pad_x = 0.01
#     pad_y = 0.01

#     rect = Rectangle(
#         (x0 - pad_x, y0 - pad_y),
#         (x1 - x0) + 2 * pad_x,
#         (y1 - y0) + 2 * pad_y,
#         transform=fig.transFigure,
#         fill=False, linewidth=1.5, edgecolor='black', zorder=10
#     )
#     fig.add_artist(rect)

#     # Optional: add a clean column title above the box (bigger than subplot titles)
#     # fig.text((x0 + x1) / 2, y1 + 0.012, Models[i],
#     #          ha='center', va='bottom', fontsize=12, fontweight='bold')

# Save before show to keep legend in file; include bbox_inches="tight"
plt.savefig(saveFilename, dpi=300, bbox_inches="tight")
plt.show()