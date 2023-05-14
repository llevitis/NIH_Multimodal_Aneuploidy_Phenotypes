import pandas as pd 
from enigmatoolbox.cross_disorder import cross_disorder_effect
from enigmatoolbox.plotting import plot_cortical, plot_subcortical
from enigmatoolbox.utils import parcel_to_surface, surface_to_parcel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import nibabel as nib

from sklearn.decomposition import PCA
import statsmodels.formula.api as smf

import sys
import importlib
sys.path.insert(0, "../../code/utils/")

import utils, spintest_utils, enigma_utils
importlib.reload(utils)
importlib.reload(spintest_utils)
importlib.reload(enigma_utils)

from enigmatoolbox.datasets.base import load_summary_stats
from enigmatoolbox.permutation_testing import spin_test, shuf_test


# Set the matplotlib font

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.family'] = 'Lato'
sns.set_context('notebook', font_scale=1.5)

# Get the HCP ROIs

with open('../../data/hcp_rois.txt', 'r') as f: 
    hcp_rois_orig = f.read().split("\n")[:-1]

# Phenotype selected
    
# need to change some '_' to '-' to match with proper HCP naming convention
hcp_rois = []
for roi in hcp_rois_orig:
    for roi2 in ['p9_46v', 'a9_46v', '9_46d', 'i6_8', 's6_8', 'OP2_3']: 
        if roi2 in roi:
            roi2_new = roi2.replace("_", "-")
            roi = roi.replace(roi2, roi2_new)
    hcp_rois.append(roi)

# Set the ENIGMA parameters

measure = ['CortSurf', 'CortThick']
phenotype = "".join(measure)
ignore = ['mega', 'medicatedcase','typeI_vs_typeII', 'early_vs_late', 'firstepisode_vs_recurrent', 'late',
          'gge', 'ltle', 'rtle', 'allotherepilepsy', 'allages']
disorder_all = ['22q', 'asd', 'adhd', 'bipolar', 'depression', 'epilepsy', 'ocd', 'schizophrenia']
disorder_behav = ['asd', 'bipolar', 'adhd', 'depression', 'epilepsy', 'ocd', 'schizophrenia']
disorder_behav_no_epilepsy = ['asd', 'adhd', 'bipolar', 'depression','ocd', 'schizophrenia']
include = ['case_vs_controls', 'adult', 'meta']

names_behav, mat_d_behav = enigma_utils.generate_enigma_pca_input(measure, ignore, disorder_behav, include)
names_behav_NE, mat_d_behav_NE = enigma_utils.generate_enigma_pca_input(measure, ignore, disorder_behav_no_epilepsy, include)

## Create a dataframe and plot a heatmap with the DK ROI by cortical map matrix
names_behav_cortex = names_behav['cortex']
mat_d_behav_cortex = mat_d_behav['cortex']

names_behav_NE_cortex = names_behav_NE['cortex']
mat_d_behav_NE_cortex = mat_d_behav_NE['cortex']

#enigma_behavNE_df = pd.DataFrame(data=mat_d_behav_NE['cortex'].T, index=names_behav_NE['cortex'])
enigma_behav_cortex_df = pd.DataFrame(data=mat_d_behav_NE_cortex, index=names_behav_NE_cortex)
plt.figure(figsize=(12,10))
g = sns.heatmap(enigma_behav_cortex_df,
                cmap='RdBu_r',
                xticklabels=False, yticklabels=True,
                cbar_kws={'label': 'Standardized \u03B2 coefficient'})
g.set_xlabel("DKT brain regions")
filepath = f"../../figures/enigma_behavDisorders_{phenotype}_heatmap.png"
plt.savefig(filepath, transparent=True,
            dpi=300, bbox_inches='tight')


full_disorder_names = {}
#full_disorder_names['all'] = names_all_cortex
full_disorder_names['behav'] = names_behav_cortex
full_disorder_names['behav_NE'] = names_behav_NE_cortex

# Run PCA 
enigma_pca_dict = {'behav': {}, 'behav_NE': {}}
for analysis in ['behav', 'behav_NE']: 
    if analysis == 'behav': 
        pca_res, variance, feature_weights = enigma_utils.run_pca(mat_d_behav_cortex)
    elif analysis == 'behav_NE': 
        pca_res, variance, feature_weights = enigma_utils.run_pca(mat_d_behav_NE_cortex)
    enigma_pca_dict[analysis]['scores'] = pca_res 
    enigma_pca_dict[analysis]['feature_weights'] = feature_weights
    enigma_pca_dict[analysis]['variance'] = variance

# Create dataframe for the feature weights of the first principal component
enigma_pca_featureDF_dict = {}
for analysis in ['behav', 'behav_NE']:  
    enigma_pca_featureDF_dict[analysis] = enigma_utils.create_pc_feature_weights_df(full_disorder_names[analysis], enigma_pca_dict[analysis]['feature_weights'])
    enigma_pca_featureDF_dict[analysis]['PC'] = enigma_pca_featureDF_dict[analysis].index

# Plot the distribution of feature weights onto PC1
for analysis in ['behav', 'behav_NE']:
    enigma_utils.plot_pc1_feature_weights(enigma_pca_featureDF_dict[analysis], full_disorder_names[analysis], f"../../figures/enigma_pc1_featureweights_{analysis}_{phenotype}.png")

# Get the aneuploidy data 

scd_bcoef_dkt_data = pd.read_csv("../../data/aneuploidy_dkt_pc1_avg.csv", index_col=0)

p_enigmaBehav, d_enigmaBehav = spin_test(scd_bcoef_dkt_data['avg'], enigma_pca_dict['behav']['scores'][:,0],
                             surface_name='fsa5', parcellation_name='aparc',
                             type='pearson', n_rot=1000, null_dist=True)

p_enigmaBehavNE, d_enigmaBehavNE = spin_test(scd_bcoef_dkt_data['avg'], enigma_pca_dict['behav_NE']['scores'][:,0],
                             surface_name='fsa5', parcellation_name='aparc',
                             type='pearson', n_rot=1000, null_dist=True)

# p_enigmaAll, d_enigmaAll = spin_test(scd_bcoef_dkt_data['avg'], enigma_pca_dict['all']['scores'][:,0],
#                              surface_name='fsa5', parcellation_name='aparc',
#                              type='pearson', n_rot=1000, null_dist=True)

types = ['behavDisorders', 'behavNEDisorders']
for i, enigma_map in enumerate([enigma_pca_dict['behav']['scores'][:,0], enigma_pca_dict['behav_NE']['scores'][:,0]]):
    plt.figure(figsize=(6,6))
    g = sns.regplot(scd_bcoef_dkt_data['avg'], enigma_map, color='black', scatter_kws={'s':20})
    r,p = stats.pearsonr(scd_bcoef_dkt_data['avg'], enigma_map)
    if i==0: 
        p_spin = p_enigmaBehav
    elif i==1:
        p_spin = p_enigmaBehavNE
    g.set_xlabel("Aneuploidy avg multimodal PC1")
    g.set_ylabel("ENIGMA cross-disorder PC1")
    #g.set_xlim([-1.5,1.5])
    g.set_ylim([-1.5,g.get_ylim()[1]+.5])
    g.text(x=-.7, y=g.get_ylim()[1]-.3, s=f'r = {np.round(r,2)} ' + '$P_{SPIN}$ =' + f' {p_spin}')
    sns.despine()
    plt.savefig(f"../../figures/aneuploidy_enigma-{types[i]}_{phenotype}_dkt_corr.png", dpi=300, bbox_inches='tight', transparent=True)


# Prepare ENIGMA & aneuploidy cortical maps for plotting using PySurfer 

from enigmatoolbox.datasets import load_summary_stats

# Load summary statistics for ENIGMA-22q
sum_stats = load_summary_stats('22q')
# Get case-control cortical thickness and surface area tables
CT = sum_stats['CortThick_case_vs_controls']

enigma_dkt_pc1_df = pd.DataFrame(index=CT.Structure, columns=['BehavDisordersPC1', 'BehavNEDisordersPC1','AneuploidyPC1'])
enigma_dkt_pc1_df.loc[:,'BehavDisordersPC1'] = enigma_pca_dict['behav']['scores'][:,0]
enigma_dkt_pc1_df.loc[:,'BehavNEDisordersPC1'] = enigma_pca_dict['behav_NE']['scores'][:,0]
enigma_dkt_pc1_df.loc[:,'AneuploidyPC1'] = list(scd_bcoef_dkt_data['avg'])

enigma_dkt_pc1_df.to_csv(f"../../data/enigma_dkt_pc1_{phenotype}_df.csv")

# Plot a correlation heatmap for the average aneuploidy PC1 map + the ENIGMA cortical maps
enigma_behav_cortex_df.loc['pc1_avg',:] = scd_bcoef_dkt_data['avg']

corr = enigma_behav_cortex_df.T.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(corr, mask=mask, square=True, xticklabels=True, yticklabels=True, cmap='RdBu_r', center=0)
plt.savefig(f"../../figures/aneuploidy_enigma_{phenotype}_heatmap.png", dpi=300, transparent=True,
            bbox_inches='tight')