import importlib
import os
import json
import sys
import random
from functools import reduce 
from collections import Counter
from glob import glob
sys.path.insert(0, "../../code/analysis_code/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from adjustText import adjust_text
from matplotlib.lines import Line2D
import seaborn as sns
import plotly.graph_objects as go
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib 
import nilearn.plotting as plotting
from sklearn.decomposition import PCA
import utils
import spintest_utils
importlib.reload(utils)
importlib.reload(spintest_utils)
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
import mat73 
from pathlib import Path

"""
Instantiate color palette
"""
paired_pal = sns.color_palette('Paired',6).as_hex()

"""
Plot the PC1 score comparison across the aneuploidies
"""

def plot_pca_dataset_score_comparisons(data, dataset_pairs, filepath=""): 
    sns.set_context('notebook', font_scale=2)
    fig, axes = plt.subplots(1,len(dataset_pairs)) 
    fig.set_figheight(5)
    fig.set_figwidth(20)
    
    for i, dp in enumerate(dataset_pairs): 
        d1 = dp[0].split('_')[0]
        d2 = dp[1].split('_')[0]
        r,p = stats.pearsonr(data[dp[0]][:,0], data[dp[1]][:,0])
        dp1_dp2_df = pd.DataFrame(columns=[d1,d2])
        dp1_dp2_df[d1] = data[dp[0]][:,0]
        dp1_dp2_df[d2] = data[dp[1]][:,0]
        g = sns.regplot(x=d1, y=d2, data=dp1_dp2_df, ax=axes[i],  color='grey')
        sns.despine()
        g.set_xlim([-1.5,2])
        g.set_ylim([-2,3.5])
        g.text(s=f"R = {np.round(r,2)}", x=-1.25, y=3, fontsize=26)
        g.set_xlabel(d1.upper())
        g.set_ylabel(d2.upper())
    plt.savefig(filepath, dpi=300, transparent=True, bbox_inches='tight')


"""
Plot the PCA feature comparison across the aneuploidies
"""

def plot_pca_dataset_feature_comparisons(data, dataset_pairs, filepath=""): 
    sns.set_context('notebook', font_scale=2)
    fig, axes = plt.subplots(1,len(dataset_pairs)) 
    fig.set_figheight(5)
    fig.set_figwidth(20)
    for i, dp in enumerate(dataset_pairs): 
        d1 = dp[0].split('_')[0]
        d2 = dp[1].split('_')[0]
        axes[i].scatter(data[dp[0]][0,:], data[dp[1]][0,:], color='grey', s=30)
        r,p = stats.pearsonr(data[dp[0]][0,:], data[dp[1]][0,:])
        sns.despine()
        axes[i].set_xlim([-0.6,0.5])
        axes[i].set_ylim([-0.6,0.5])
        axes[i].text(s=f"R = {np.round(r,2)}", x=-.5, y=0.4, fontsize=26)
        axes[i].set_xlabel(d1.upper())
        axes[i].set_ylabel(d2.upper())
        texts = []
        for j, txt in enumerate(t1w_dti_func_features):
            texts.append(axes[i].text(data[dp[0]][0,j], 
                                      data[dp[1]][0,j],
                                      txt, fontsize=22))
        adjust_text(texts, ax=axes[i])
    plt.savefig(filepath, dpi=300, transparent=True, bbox_inches='tight')    

"""
Plot variance explained for the PCA
"""
def plot_PCA_varExplained(data, filepath=""):
    plt.figure(figsize=(6,8))
    g = sns.pointplot(data=data, 
                x='variable', 
                y='value', 
                hue='Dataset', 
                palette={'XXY': paired_pal[3], 'XYY': paired_pal[4], 'T21': paired_pal[5]})
    sns.despine()
    plt.xticks(rotation=45)
    plt.yticks([0,5,10,15,20,25,30,35,40])
    plt.ylabel("Variance Explained %")
    plt.xlabel("")
    g.get_legend().remove()
    #g.legend_.set_title('Disorder')
    # replace labels
    #new_labels = ['XXY']
    #for t, l in zip(g.legend_.texts, new_labels):
    #    t.set_text(l)
    plt.legend(fancybox=True)
    plt.savefig(filepath, transparent=True, 
                bbox_inches='tight', dpi=300)
        
"""
Plot correlation matrix for scores or features across all aneuploidies
"""

def plot_corrMatrix_PCA(data, title="", filepath=""):

    plt.figure(figsize=(12,12))
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(data.corr(), dtype=bool))

    ax = sns.heatmap(data.corr(), cmap='coolwarm',
                    mask=mask,
                    center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5, 'label': 'Correlation'},
                    vmin=-1, vmax=1, xticklabels=True, yticklabels=True)

    labels = ["_".join([x.split("_")[0],x.split("_")[2]]) for x in data.columns]

    ax.set_title(title, fontsize=24)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.figure.axes[-1].yaxis.label.set_size(30)
    plt.savefig(filepath, 
                transparent=True, 
                dpi=300, 
                bbox_inches='tight')
    

"""
Plot ANOVA results for PC1 scores and specific annotations
"""
def plot_anova_results(data, disorders, annotations, ylabels, spins, filepath):
    annotation_df = pd.DataFrame(index=disorders, columns=annotations)
    fig, axes = plt.subplots(len(disorders),len(annotations), 
                             figsize=(len(disorders)*2.5,len(annotations)*8), sharey=True, sharex=True)
    for i, disorder in enumerate(disorders):
        j = 0
        for annotation in annotations:
            annotation_name = annotation
            real_fstat = spintest_utils.compute_real_Fstat(data, disorder, annotation)
            null_fstats = spintest_utils.compute_null_Fstat(data, spins, disorder, annotation)
            pval = len([x for x in null_fstats if x > real_fstat]) / len(null_fstats)
            #print(disorder, annotation, len([x for x in null_fstats if x > real_fstat]))
            annotation_df.loc[disorder, annotation] = pval
            sns.histplot(null_fstats, label='Null', ax=axes[i,j], color='red')
            sns.despine()
            axes[i,j].axvline(real_fstat, color='black', label='Empirical')
            if i == 0:
                axes[i,j].set_title(f'{annotation_name}', fontsize=24)
                if j == 1:      
                    axes[i,j].legend(bbox_to_anchor=(1,1))
            axes[i,j].text(x=18, y=75, s=f'p = {pval}')
            #plt.show()
            j+=1
    for i, ylabel in enumerate(ylabels):
        axes[i,0].set_ylabel(ylabel, fontsize=24)
    fig.text(0.65, 0.05, 'F statistic', ha='center', fontsize=24)
    fig.suptitle('Empirical vs Null F-statistics\n Across Disorders & Annotations', fontsize=28) # or plt.suptitle('Main title')
    plt.savefig(filepath, transparent=True, bbox_inches='tight', dpi=300)



"""
Load data
"""

scd_bcoef_effect_files = glob('../../dummy_data/*scd_bcoef_effect_*.csv')

scd_bcoef_effect_dict = {} 
for file in scd_bcoef_effect_files:
    key = file.split("/")[-1].split("_")[0] + '_' + file.split("/")[-1].split("_")[4].split(".")[0]
    scd_bcoef_effect_dict[key] = pd.read_csv(file, index_col=0)

func_rois_to_keep = list(scd_bcoef_effect_dict['xxy_withoutTTV'].index)
t1w_dti_func_features = ['thickness', 'meancurv', 'volume', 'area', 'curvind', 'foldind', 
                         'gauscurv', 'fa', 'md', 'ad', 'rd', 'ga', 'mode', 
                         'reho', 'alff']
t1w_dti_func_bcoef_features = [x + '_bcoef' for x in t1w_dti_func_features]

with open('../../data/hcp_rois.txt', 'r') as f: 
    hcp_rois = f.read().split("\n")[:-1]

# Load the pre-computed spins 
spins = np.load('../../data/spins_hcp.npy')

# Instantiate dictionaries to hold PCA scores, variance explained, and feature weights
# This is done for delta IDP matrices computed both with and without TTV correction.

scd_bcoef_PCAscores_dict = {}
scd_bcoef_PCAvarexplained_dict = {}
scd_bcoef_PCAfeatureweights_dict = {}

for key in scd_bcoef_effect_dict.keys(): 
    pca = PCA(n_components=10)
    pca_res = pca.fit_transform(scd_bcoef_effect_dict[key][t1w_dti_func_bcoef_features])
    scd_bcoef_PCAscores_dict[key] = pca_res
    scd_bcoef_PCAvarexplained_dict[key] = pca.explained_variance_ratio_
    scd_bcoef_PCAfeatureweights_dict[key] = pca.components_

# Plot comparisons across scores and features 

plot_pca_dataset_score_comparisons(data=scd_bcoef_PCAscores_dict,
                                   dataset_pairs=[['xxy_withTTV', 'xyy_withTTV'],['xxy_withTTV', 't21_withTTV'], ['xyy_withTTV', 't21_withTTV']], 
                                   filepath="../../dummy_figures/allAneuploidies_pc1ScoreComparison_withTTV.png")

plot_pca_dataset_score_comparisons(data=scd_bcoef_PCAscores_dict,
                                   dataset_pairs=[['xxy_withoutTTV', 'xyy_withoutTTV'],['xxy_withoutTTV', 't21_withoutTTV'], ['xyy_withoutTTV', 't21_withoutTTV']], 
                                   filepath="../../dummy_figures/allAneuploidies_pc1ScoreComparison_withoutTTV.png")

plot_pca_dataset_feature_comparisons(data=scd_bcoef_PCAfeatureweights_dict,
                                   dataset_pairs=[['xxy_withTTV', 'xyy_withTTV'],['xxy_withTTV', 't21_withTTV'], ['xyy_withTTV', 't21_withTTV']], 
                                   filepath="../../dummy_figures/allAneuploidies_pc1FeatureComparison_withTTV.png")

plot_pca_dataset_feature_comparisons(data=scd_bcoef_PCAfeatureweights_dict,
                                   dataset_pairs=[['xxy_withoutTTV', 'xyy_withoutTTV'],['xxy_withoutTTV', 't21_withoutTTV'], ['xyy_withoutTTV', 't21_withoutTTV']], 
                                   filepath="../../dummy_figures/allAneuploidies_pc1FeatureComparison_withoutTTV.png")

"""
Plot the PCA variance explained
""" 

scd_bcoef_PCAvarexplained_df = pd.DataFrame(columns=[f'PC{i}' for i in range(1,11)])
for key in scd_bcoef_PCAvarexplained_dict.keys(): 
    for i,pc in enumerate([f'PC{i}' for i in range(1,11)]):
        scd_bcoef_PCAvarexplained_df.loc[key, pc] = scd_bcoef_PCAvarexplained_dict[key][i]
    scd_bcoef_PCAvarexplained_df.loc[key, 'Dataset'] = key


scd_bcoef_PCAvarexplained_melted_df = scd_bcoef_PCAvarexplained_df.melt(id_vars=['Dataset'])
scd_bcoef_PCAvarexplained_melted_df['TTV_Status'] = [x.split("_")[1] for x in scd_bcoef_PCAvarexplained_melted_df['Dataset']]
scd_bcoef_PCAvarexplained_melted_df['Dataset'] = [x.split("_")[0] for x in scd_bcoef_PCAvarexplained_melted_df['Dataset']]
scd_bcoef_PCAvarexplained_melted_df['value'] *= 100

scd_bcoef_PCAvarexplained_melted_df['Dataset'] = scd_bcoef_PCAvarexplained_melted_df['Dataset'].map({'xxy': 'XXY', 'xyy': 'XYY', 't21': 'T21'})


for ttv_stat in ['withTTV', 'withoutTTV']:
    plot_PCA_varExplained(data=scd_bcoef_PCAvarexplained_melted_df[scd_bcoef_PCAvarexplained_melted_df['TTV_Status']==ttv_stat], 
                          filepath=f"../../dummy_figures/pca_bcoef_{ttv_stat}.png")

"""
Supplemental plots for correlation across all PC1 scores and features
"""

scd_bcoef_PCAscores_reorganized_dict = {}
for key in scd_bcoef_PCAscores_dict.keys(): 
    curr_df = pd.DataFrame(index=func_rois_to_keep)
    for i in range(0, scd_bcoef_PCAscores_dict[key].shape[1]): 
        curr_df.loc[func_rois_to_keep, f'PC{i+1}'] = scd_bcoef_PCAscores_dict[key][:,i]
    scd_bcoef_PCAscores_reorganized_dict[key] = curr_df

scd_pc_features_1to5_withTTV_df = pd.DataFrame(index=t1w_dti_func_features)
for key in ['xxy_withTTV', 'xyy_withTTV', 't21_withTTV']: 
    for i,pc in enumerate(['PC1', 'PC2', 'PC3', 'PC4', 'PC5']):
        aneuploidy = key.split("_")[0]
        scd_pc_features_1to5_withTTV_df.loc[:, f'{key}_{pc}'] = scd_bcoef_PCAfeatureweights_dict[key][i,:]
        
scd_pc_features_1to5_withoutTTV_df = pd.DataFrame(index=t1w_dti_func_features)
for key in ['xxy_withoutTTV', 'xyy_withoutTTV', 't21_withoutTTV']: 
    for i,pc in enumerate(['PC1', 'PC2', 'PC3', 'PC4', 'PC5']):
        aneuploidy = key.split("_")[0]
        scd_pc_features_1to5_withoutTTV_df.loc[:, f'{key}_{pc}'] = scd_bcoef_PCAfeatureweights_dict[key][i,:]

scd_pc_scores_1to5_withTTV_df = pd.DataFrame(index=func_rois_to_keep)
for key in ['xxy_withTTV', 'xyy_withTTV', 't21_withTTV']: 
    for pc in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
        aneuploidy = key.split("_")[0]
        scd_pc_scores_1to5_withTTV_df.loc[:, f'{key}_{pc}'] = scd_bcoef_PCAscores_reorganized_dict[key][pc]
        
scd_pc_scores_1to5_withoutTTV_df = pd.DataFrame(index=func_rois_to_keep)
for key in ['xxy_withoutTTV', 'xyy_withoutTTV', 't21_withoutTTV']: 
    for pc in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
        aneuploidy = key.split("_")[0]
        scd_pc_scores_1to5_withoutTTV_df.loc[:, f'{key}_{pc}'] = scd_bcoef_PCAscores_reorganized_dict[key][pc]   

scd_pc_features_1to5_withWithoutTTV_df = pd.concat([scd_pc_features_1to5_withTTV_df, 
                                                    scd_pc_features_1to5_withoutTTV_df], 
                                                    axis=1)

scd_pc_scores_1to5_withWithoutTTV_df = pd.concat([scd_pc_scores_1to5_withTTV_df, 
                                                  scd_pc_scores_1to5_withoutTTV_df], 
                                                  axis=1)

plot_corrMatrix_PCA(scd_pc_features_1to5_withTTV_df, 
                    title="PC Features (With TTV)", 
                    filepath="../../dummy_figures/allAneuploidies_PC1-5_features_corrMatrix_withTTV.png")
plot_corrMatrix_PCA(scd_pc_scores_1to5_withTTV_df, 
                    title="PC Scores (Without TTV)", 
                    filepath="../../dummy_figures/allAneuploidies_PC1-5_scores_corrMatrix_withTTV.png")
plot_corrMatrix_PCA(scd_pc_features_1to5_withWithoutTTV_df, 
                    title="PC Features (With vs Without TTV)", 
                    filepath="../../dummy_figures/allAneuploidies_PC1-5_features_corrMatrix_withWithoutTTV.png")
plot_corrMatrix_PCA(scd_pc_scores_1to5_withWithoutTTV_df, 
                    title="PC Scores (With vs Without TTV)", 
                    filepath="../../dummy_figures/allAneuploidies_PC1-5_scores_corrMatrix_withWithoutTTV.png")

"""
Annotation using the Yeo-Krienen-17 and the Von Economo-Koskinas atlases
"""

# load in the Von Economo atlas mapping
vEatlas = mat73.loadmat('../../data/vE_atlas.mat')

hcp_yeo_vonEconomo_df = pd.read_csv("../../data/glasser_yeo7_yeo17_vonEconomo_map.csv")
yeo17networks = pd.read_csv("../../data/yeo17names.csv")
yeo17networks['YeoNetwork'] = [x.rstrip() for x in yeo17networks['YeoNetwork']]
yeo17networks['YeoNetworkFullName'] = [x.rstrip() for x in yeo17networks['YeoNetworkFullName']]

vE_ORDER = [
    'prim_motor',
    'assoc_1',
    'assoc_2',
    'sec_sensory',
    'prim_sensory',
    'limbic',
    'insula'
]


vE_CODES = {
    'prim_motor': 1,
    'assoc_1': 2,
    'assoc_2': 3,
    'sec_sensory': 4,
    'prim_sensory': 5,
    'limbic': 6,
    'insula': 7
}

YEO17_ORDER = [
    'MOT-1',
    'MOT-2', 
    'MOT-3', 
    'VIS-1',  
    'VIS-2', 
    'DAN-1',
    'DAN-2',
    'FP-1',
    'FP-2', 
    'FP-3',
    'FP-4',
    'LIM-1',
    'LIM-2',
    'DMN-1',
    'DMN-2',
    'DMN-3',
    'VAN-1'
]

YEO17_CODES = {
    'VIS-1': 1,
    'VIS-2': 2,
    'MOT-1': 3,
    'MOT-2': 4,
    'DAN-2': 5,
    'DAN-1': 6,
    'VAN-1': 7,
    'FP-1': 8,
    'LIM-1': 9, 
    'LIM-2': 10, 
    'FP-2': 11,
    'FP-3': 12, 
    'FP-4': 13, 
    'MOT-3': 14, 
    'DMN-3': 15, 
    'DMN-1': 16, 
    'DMN-2': 17
}

scd_bcoef_allDisorders_df = pd.DataFrame(index=func_rois_to_keep)

for disorder in ['xxy', 'xyy', 't21']: 
    scd_bcoef_allDisorders_df.loc[func_rois_to_keep, f'{disorder}_withTTV'] = scd_bcoef_PCAscores_dict[f'{disorder}_withTTV'][:,0]
    scd_bcoef_allDisorders_df.loc[func_rois_to_keep, f'{disorder}_withoutTTV'] = scd_bcoef_PCAscores_dict[f'{disorder}_withoutTTV'][:,0]
for ttv_stat in ['withTTV', 'withoutTTV']: 
    scd_bcoef_allDisorders_df.loc[func_rois_to_keep, f'avgPC1_{ttv_stat}'] = scd_bcoef_allDisorders_df[[f'xxy_{ttv_stat}',
                                                                                                        f'xyy_{ttv_stat}',
                                                                                                        f't21_{ttv_stat}']].mean(axis=1)
scd_bcoef_allDisorders_df = scd_bcoef_allDisorders_df.astype('float64')

for i, roi in enumerate(hcp_rois): 
    yeo_roi = hcp_yeo_vonEconomo_df.index[i]
    if roi in func_rois_to_keep: 
        scd_bcoef_allDisorders_df.loc[roi, 'vE'] = vEatlas['vE_atlas'][i]
        yeo17_id = hcp_yeo_vonEconomo_df.loc[yeo_roi,'Yeo17']
        scd_bcoef_allDisorders_df.loc[roi, 'Yeo17'] = yeo17_id
        scd_bcoef_allDisorders_df.loc[roi, 'Yeo17Label'] = yeo17networks[yeo17networks.ID==yeo17_id]['YeoNetwork'].values[0]

for ttv_stat in ['withTTV', 'withoutTTV']:
    plot_anova_results(data=scd_bcoef_allDisorders_df, 
                       disorders=[f'xxy_{ttv_stat}', f'xyy_{ttv_stat}', f't21_{ttv_stat}', f'avgPC1_{ttv_stat}'], 
                       annotations=['Yeo17', 'vE'], 
                       ylabels=['XXY PC1', 'XYY PC1', 'T21 PC1', 'Avg PC1'], 
                       spins=spins,
                       filepath=f'../../dummy_figures/supp_anova_annot_{ttv_stat}_yeo17-vE')

# Create the dataframes storing information about PC1 enrichment across the annotations 
partitions_specificity_bcoef_dict = {} 
for ttv_stat in ['withTTV', 'withoutTTV']:
    partitions_specificity_bcoef_dict[f'vE_{ttv_stat}'] = spintest_utils.make_partitions_specificity_df(data=scd_bcoef_allDisorders_df, 
                                                                                                        disorders=[f'xxy_{ttv_stat}', f'xyy_{ttv_stat}', f't21_{ttv_stat}', f'avgPC1_{ttv_stat}'], 
                                                                                                        spins=spins,
                                                                                                        annotation_name='vE', 
                                                                                                        annotation_codes=vE_CODES,
                                                                                                        annotation_order=vE_ORDER)
    partitions_specificity_bcoef_dict[f'Yeo17_{ttv_stat}'] = spintest_utils.make_partitions_specificity_df(data=scd_bcoef_allDisorders_df, 
                                                                                                        disorders=[f'xxy_{ttv_stat}', f'xyy_{ttv_stat}', f't21_{ttv_stat}', f'avgPC1_{ttv_stat}'], 
                                                                                                        spins=spins,
                                                                                                        annotation_name='Yeo17', 
                                                                                                        annotation_codes=YEO17_CODES,
                                                                                                        annotation_order=YEO17_ORDER)

vE_paired_pal = sns.color_palette('Dark2',len(vE_ORDER)).as_hex()
vE_paired_pal_dict = {}
for i in range(0, len(vE_ORDER)):
    flipped_dict = {value:key for key, value in vE_CODES.items()}
    network = flipped_dict[i+1]
    vE_paired_pal_dict[network] = vE_paired_pal[i] 


yeo17_colorLUT = pd.read_csv('../../data/Yeo2011_17Networks_ColorLUT.tsv', sep='\t')
for i in yeo17_colorLUT.index: 
    r = yeo17_colorLUT.loc[i, 'R']
    g = yeo17_colorLUT.loc[i, 'G']
    b = yeo17_colorLUT.loc[i, 'B'] 
    curr_hex = utils.rgb2hex(r,g,b)
    yeo17_colorLUT.loc[i, 'hexcode'] = curr_hex
    yeo17_colorLUT.loc[i, 'YeoNetwork'] = yeo17networks.loc[i, 'YeoNetwork'].rstrip().lstrip()

yeo17_paired_pal_dict = {}
for i in range(0, len(YEO17_ORDER)):
    flipped_dict = {value:key for key, value in YEO17_CODES.items()}
    network = flipped_dict[i+1]
    yeo17_paired_pal_dict[network] = yeo17_colorLUT[yeo17_colorLUT.YeoNetwork==network]['hexcode'].values[0]



for ttv_stat in ['withTTV', 'withoutTTV']:
    utils.make_barplot_horizontal(data=partitions_specificity_bcoef_dict[f'vE_{ttv_stat}'], 
                                  netorder=list(partitions_specificity_bcoef_dict[f'vE_{ttv_stat}'].query("disorder == 'avgPC1'").sort_values(by='zscore')['network']), 
                                  diff_colorbars=vE_paired_pal_dict,
                                  xticklabels=vE_ORDER, 
                                  disorders=['xxy','xyy','t21', 'avgPC1'], 
                                  fname=f'../../dummy_figures/cross_disorder_vE_partitions_sig_{ttv_stat}.png')
    utils.make_barplot_horizontal(data=partitions_specificity_bcoef_dict[f'Yeo17_{ttv_stat}'], 
                                  netorder=list(partitions_specificity_bcoef_dict[f'Yeo17_{ttv_stat}'].query("disorder == 'avgPC1'").sort_values(by='zscore')['network']), 
                                  diff_colorbars=yeo17_paired_pal_dict,
                                  xticklabels=YEO17_ORDER, 
                                  disorders=['xxy','xyy','t21', 'avgPC1'], 
                                  figsize=((3.125) * 4, 4),
                                  fname=f'../../dummy_figures/cross_disorder_yeo17_partitions_sig_{ttv_stat}.png')