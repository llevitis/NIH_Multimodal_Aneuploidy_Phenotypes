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
#import cmasher as cmr
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
from pathlib import Path
import networkx as nx

"""
Code for hierarchical clustering plot
"""
def plot_hierarchCluster_spin(data, pvals, disorder_color_pal, modality_color_pal, disorder_phenotype_color_dict, filepath):
    sns.set_context('notebook', font_scale=1)
    plt.figure(figsize=(10,10))
    kws = dict(cbar_kws=dict(ticks=[-1.0, -.5, 0, 0.5, 1.0]))

    data_corr = data.corr()
    g = sns.clustermap(data_corr, cmap='RdBu_r',
                       row_colors=[list(disorder_color_pal.values()), list(modality_color_pal.values())],
                       col_colors=[list(disorder_color_pal.values()), list(modality_color_pal.values())],
                       yticklabels=True,xticklabels=True,
                       vmin = -1.2, vmax=1.2, linewidths=0,
                       **kws)



    # Here labels on the y-axis are rotated
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)

    # Here we add asterisks onto cells with signficant correlations
    for i, ix in enumerate(g.dendrogram_row.reordered_ind):
        for j, jx in enumerate(g.dendrogram_row.reordered_ind):
            if i != j:
                text = g.ax_heatmap.text(
                    j + 0.32,
                    i + 0.5,
                    "*" if pvals[ix,jx] < 0.05 else "",
                    ha="center",
                    va="center",
                    color="black",
                )
                text.set_fontsize(12)

    x0, _y0, _w, _h = g.cbar_pos
    # g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.02])

    g.ax_cbar.set_title('Spatial correlation')
    g.ax_cbar.tick_params(axis='x', length=3)


    # Draw the legend bar for the classes                 
    for legend_item in list(disorder_phenotype_color_dict.keys()):
        g.ax_col_dendrogram.bar(0, 0, color=disorder_phenotype_color_dict[legend_item],
                                label=legend_item, linewidth=0)
    # g.ax_col_dendrogram.set_position([0, .9, g.ax_col_dendrogram.get_position().width, .45])
    g.cax.set_position([1, .2, .03, .45])
    g.ax_col_dendrogram.legend(ncol=2, loc='best', bbox_to_anchor=(1.35,0.55))

    plt.savefig(filepath, 
                transparent=True, 
                dpi=300, 
                bbox_inches='tight')
    

"""
Code for force directed edge graph representation of correlation matrix
"""
#fig, ax = plt.subplots(figsize=(8, 8))

# Transform it in a links data frame (3 columns only):
def plot_hierarchClust_edgeGraph(data, spins, palette, filepath): 
    t1w_features = ['thickness', 'volume', 'area', 'gauscurv', 'meancurv', 'curvind', 'foldind']
    dti_features = ['ad', 'md', 'rd', 'fa', 'ga']
    func_features = ['reho', 'alff']

    corr_matrix = data.corr()
    links = corr_matrix.stack().reset_index()
    links.columns = ['var1', 'var2', 'weight']
    links['weight'] = np.abs(links['weight'])
    links['weight'] = [x if x > np.percentile(np.abs(corr_matrix), 80) else 0 for x in links['weight']]

    for i in links.index:
        var1 = links.loc[i, 'var1']
        var2 = links.loc[i, 'var2']
        links.loc[i, 'pval'] = spins.loc[var1, var2]
        if links.loc[i, 'weight'] > 0:
            if links.loc[i, 'pval'] < 0.05:
                links.loc[i, 'color'] = 'black'
                links.loc[i, 'width'] = 1.6
            else: 
                links.loc[i, 'color'] = 'grey'
                links.loc[i, 'width'] = .6
        else:
            links.loc[i, 'color'] = 'grey'
            links.loc[i, 'width'] = 0

        # if links.loc[i, 'pval'] >= 0.05:    
        #     links.loc[i, 'weight'] = 0
        # else:
        #     links.loc[i,'weight'] = 1

    # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
    #links=links.loc[(links['var1'] != links['var2']) & (links['pval'] < 0.05)]
    links=links.loc[(links['var1'] != links['var2']) ]

    # Build your graph
    # G=nx.from_pandas_edgelist(links, 'var1', 'var2', ["weight", "pval"])
    # pos=nx.spring_layout(G)

    G=nx.from_pandas_edgelist(links, 'var1', 'var2', ["weight", "pval", "color", "width"])
    pos=nx.spring_layout(G, k=.4)
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())


    node_colors = []
    for node in G: 
        phenotype = node.split("_")[1]
        if phenotype in t1w_features:
            node_colors.append(paired_pal[0])
        elif phenotype in dti_features:
            node_colors.append(paired_pal[1])
        elif phenotype in func_features:
            node_colors.append(paired_pal[2])

    edge_colors = []
    for node in G:
        aneuploidy = node.split("_")[0]
        if aneuploidy == 'xxy':
            edge_colors.append(palette[3])
        elif aneuploidy == 'xyy':
            edge_colors.append(palette[4])
        elif aneuploidy == 't21':
            edge_colors.append(palette[5])

    edgecolors = nx.get_edge_attributes(G,'color').values()
    edgewidths = nx.get_edge_attributes(G,'width').values()


    # Plot the network:
    #I had this list for the name corresponding too the color but different from the node name
    ColorLegend = {'sMRI': palette[0],'DWI': palette[1], 'rsFMRI': palette[2],
                   'XXY': palette[3], 'XYY': palette[4], 'T21': palette[5]}

    # Using a figure to use it as a parameter when calling nx.draw_networkx
    f = plt.figure(figsize=(6,6))
    ax = f.add_subplot(1,1,1)
    for label in ColorLegend:
        ax.plot([0],[0],color=ColorLegend[label],label=label, linewidth=2)

    nx.draw_networkx(G, pos, with_labels=False, node_color=edge_colors,edge_color=edgecolors, edgecolors=node_colors, width=list(edgewidths), node_size=80, font_size=15)

    # Setting it to how it was looking before.                                                                                                              
    plt.axis('off')
    f.set_facecolor('w')
    f.tight_layout()
    plt.savefig(filepath, dpi=300, transparent=True, bbox_inches='tight')

# Load the delta IDP files
scd_bcoef_effect_files = glob('../../dummy_data/*scd_bcoef_effect_*.csv')

scd_bcoef_effect_dict = {} 
for file in scd_bcoef_effect_files:
    key = file.split("/")[-1].split("_")[0] + '_' + file.split("/")[-1].split("_")[4].split(".")[0]
    scd_bcoef_effect_dict[key] = pd.read_csv(file, index_col=0)

func_rois_to_keep = list(scd_bcoef_effect_dict['xxy_withoutTTV'].index)
t1w_dti_func_features = ['thickness', 'meancurv', 'volume', 'area', 'curvind', 'foldind', 
                         'gauscurv', 'fa', 'md', 'ad', 'rd', 'ga', 'mode', 
                         'reho', 'alff']

with open('../../data/hcp_rois.txt', 'r') as f: 
    hcp_rois = f.read().split("\n")[:-1]

# Load the pre-computed spins 
spins = np.load('../../data/spins_hcp.npy')

# Set the color palette 
paired_pal = sns.color_palette('Paired',6).as_hex()

# Create joint cross-aneuploidy delta IDP df for TTV-corrected and TTV-uncorrected data
xxy_xyy_t21_withoutTTV_bcoef_joint_df = pd.DataFrame(index=func_rois_to_keep)
for feature in t1w_dti_func_features:
    xxy_xyy_t21_withoutTTV_bcoef_joint_df.loc[:, f'xxy_{feature}'] = scd_bcoef_effect_dict['xxy_withoutTTV'].loc[:, f'{feature}_bcoef']
    xxy_xyy_t21_withoutTTV_bcoef_joint_df.loc[:, f'xyy_{feature}'] = scd_bcoef_effect_dict['xyy_withoutTTV'].loc[:, f'{feature}_bcoef']
    xxy_xyy_t21_withoutTTV_bcoef_joint_df.loc[:, f't21_{feature}'] = scd_bcoef_effect_dict['t21_withoutTTV'].loc[:, f'{feature}_bcoef']

xxy_xyy_t21_withTTV_bcoef_joint_df = pd.DataFrame(index=func_rois_to_keep)
for feature in t1w_dti_func_features:
    xxy_xyy_t21_withTTV_bcoef_joint_df.loc[:, f'xxy_{feature}'] = scd_bcoef_effect_dict['xxy_withTTV'].loc[:, f'{feature}_bcoef']
    xxy_xyy_t21_withTTV_bcoef_joint_df.loc[:, f'xyy_{feature}'] = scd_bcoef_effect_dict['xyy_withTTV'].loc[:, f'{feature}_bcoef']
    xxy_xyy_t21_withTTV_bcoef_joint_df.loc[:, f't21_{feature}'] = scd_bcoef_effect_dict['t21_withTTV'].loc[:, f'{feature}_bcoef']

# Make the null dataframes
xxy_xyy_t21_withoutTTV_joint_spin_bcoef_null_dict = {}
for i in range(0, 1000):
    curr_null_df = pd.DataFrame(index=xxy_xyy_t21_withoutTTV_bcoef_joint_df.index, columns=xxy_xyy_t21_withoutTTV_bcoef_joint_df.columns)
    for idp in curr_null_df.columns: 
        brain = np.array(xxy_xyy_t21_withoutTTV_bcoef_joint_df.loc[:,idp])
        curr_null_df.loc[:,idp] = brain[spins[:,i]]
    xxy_xyy_t21_withoutTTV_joint_spin_bcoef_null_dict[i] = curr_null_df

xxy_xyy_t21_withTTV_joint_spin_bcoef_null_dict = {}
for i in range(0, 1000):
    curr_null_df = pd.DataFrame(index=xxy_xyy_t21_withTTV_bcoef_joint_df.index, columns=xxy_xyy_t21_withTTV_bcoef_joint_df.columns)
    for idp in curr_null_df.columns: 
        brain = np.array(xxy_xyy_t21_withTTV_bcoef_joint_df.loc[:,idp])
        curr_null_df.loc[:,idp] = brain[spins[:,i]]
    xxy_xyy_t21_withTTV_joint_spin_bcoef_null_dict[i] = curr_null_df

# Compute the permutated correlations 
permcorrs_withoutTTV_bcoef_spin = np.vstack([
        spintest_utils._get_permcorr(xxy_xyy_t21_withoutTTV_bcoef_joint_df, xxy_xyy_t21_withoutTTV_joint_spin_bcoef_null_dict[n])
        for n in range(len(xxy_xyy_t21_withoutTTV_joint_spin_bcoef_null_dict))
    ])

permcorrs_withTTV_bcoef_spin = np.vstack([
        spintest_utils._get_permcorr(xxy_xyy_t21_withTTV_bcoef_joint_df, xxy_xyy_t21_withTTV_joint_spin_bcoef_null_dict[n])
        for n in range(len(xxy_xyy_t21_withTTV_joint_spin_bcoef_null_dict))
    ])

# Compute the pvalues following the spin test

pvals_withoutTTV_bcoef_spin = spintest_utils.get_fwe(np.corrcoef(np.array(xxy_xyy_t21_withoutTTV_bcoef_joint_df).T), permcorrs_withoutTTV_bcoef_spin)
pvals_withTTV_bcoef_spin = spintest_utils.get_fwe(np.corrcoef(np.array(xxy_xyy_t21_withTTV_bcoef_joint_df).T), permcorrs_withTTV_bcoef_spin)

pvals_withoutTTV_bcoef_spin_df = pd.DataFrame(index=xxy_xyy_t21_withoutTTV_bcoef_joint_df.corr().index,
                                              columns=xxy_xyy_t21_withoutTTV_bcoef_joint_df.corr().columns,
                                              data=pvals_withoutTTV_bcoef_spin)

pvals_withTTV_bcoef_spin_df = pd.DataFrame(index=xxy_xyy_t21_withTTV_bcoef_joint_df.corr().index,
                                           columns=xxy_xyy_t21_withTTV_bcoef_joint_df.corr().columns,
                                           data=pvals_withTTV_bcoef_spin)

dataset_phenotype_color_modality_dict = {}
dataset_phenotype_color_disorder_dict = {}
for dataset_phenotype in xxy_xyy_t21_withTTV_bcoef_joint_df.columns:
    if dataset_phenotype.split("_")[1] in ['thickness', 'volume', 'area', 'meancurv', 'gauscurv', 'foldind', 'curvind']:
        dataset_phenotype_color_modality_dict[dataset_phenotype] = paired_pal[0]
    elif dataset_phenotype.split("_")[1] in ['ad', 'rd', 'md', 'fa', 'ga', 'mode']:
        dataset_phenotype_color_modality_dict[dataset_phenotype] = paired_pal[1]
    else:
        dataset_phenotype_color_modality_dict[dataset_phenotype] = paired_pal[2]
        
for dataset_phenotype in xxy_xyy_t21_withTTV_bcoef_joint_df.columns:
    if dataset_phenotype.split("_")[0] == 'xxy':
        dataset_phenotype_color_disorder_dict[dataset_phenotype] = paired_pal[3]
    elif dataset_phenotype.split("_")[0] == 'xyy':
        dataset_phenotype_color_disorder_dict[dataset_phenotype] = paired_pal[4]
    else:
        dataset_phenotype_color_disorder_dict[dataset_phenotype] = paired_pal[5]

disorder_phenotype_color_dict = {'XXY': paired_pal[3], 'XYY': paired_pal[4], 'T21': paired_pal[5],
                                 'sMRI': paired_pal[0], 'DWI': paired_pal[1], 'rsFMRI': paired_pal[2]}


plot_hierarchCluster_spin(data=xxy_xyy_t21_withoutTTV_bcoef_joint_df, 
                           pvals= pvals_withoutTTV_bcoef_spin, 
                           disorder_color_pal=dataset_phenotype_color_disorder_dict,
                           modality_color_pal=dataset_phenotype_color_modality_dict, 
                           disorder_phenotype_color_dict=disorder_phenotype_color_dict,
                           filepath='../../figures/xxy_xyy_t21_hierarchClust_withoutTTV_bcoef_spin-vasa.png')

plot_hierarchCluster_spin(data=xxy_xyy_t21_withTTV_bcoef_joint_df, 
                           pvals= pvals_withTTV_bcoef_spin, 
                           disorder_color_pal=dataset_phenotype_color_disorder_dict,
                           modality_color_pal=dataset_phenotype_color_modality_dict, 
                           disorder_phenotype_color_dict=disorder_phenotype_color_dict,
                           filepath='../../figures/xxy_xyy_t21_hierarchClust_withTTV_bcoef_spin-vasa.png')

plot_hierarchClust_edgeGraph(data=xxy_xyy_t21_withTTV_bcoef_joint_df, 
                             spins=pvals_withTTV_bcoef_spin_df, 
                             palette=paired_pal, 
                             filepath='../../figures/xxy_xyy_t21_hierarchClust_withTTV_edgeGraph.png')

plot_hierarchClust_edgeGraph(data=xxy_xyy_t21_withoutTTV_bcoef_joint_df, 
                             spins=pvals_withoutTTV_bcoef_spin_df, 
                             palette=paired_pal, 
                             filepath='../../figures/xxy_xyy_t21_hierarchClust_withoutTTV_edgeGraph.png')
