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
sys.path.insert(0, "../../code/analysis_code/")

import utils
import spintest_utils
importlib.reload(utils)
importlib.reload(spintest_utils)

from enigmatoolbox.datasets.base import load_summary_stats
from enigmatoolbox.permutation_testing import spin_test, shuf_test

# Code taken from the ENIGMA toolbox sourcecode 

def generate_enigma_pca_input(measure, ignore, disorders, include): 
    mat_d = {'cortex': [], 'subcortex': []}
    names = {'cortex': [], 'subcortex': []}
    for _, ii in enumerate(disorders):
        # Load summary statistics
        sum_stats = load_summary_stats(ii)
        fieldos = list(sum_stats.keys())

        # Loop through structure fields (case-control options)
        for kk, jj in enumerate(fieldos):
            if 'Cort' in jj:
                if not include:
                    if not any(ig in jj for ig in ignore) and any(meas in jj for meas in measure):
                        mat_d['cortex'].append(sum_stats[jj].iloc[:, 2])
                        names['cortex'].append(ii + ': ' + jj)

                elif include:
                    if any(inc in jj for inc in include) and not any(ig in jj for ig in ignore) \
                            and any(meas in jj for meas in measure):
                        mat_d['cortex'].append(sum_stats[jj].iloc[:, 2])
                        names['cortex'].append(ii + ': ' + jj)

            if 'Sub' in jj:
                if not include:
                    if not any(ig in jj for ig in ignore) and any(meas in jj for meas in measure):
                        mat_d['subcortex'].append(sum_stats[jj].iloc[:, 2])
                        names['subcortex'].append(ii + ': ' + jj)

                elif include:
                    if any(inc in jj for inc in include) and not any(ig in jj for ig in ignore) \
                            and any(meas in jj for meas in measure):
                        mat_d['subcortex'].append(sum_stats[jj].iloc[:, 2])
                        names['subcortex'].append(ii + ': ' + jj)

    for ii, jj in enumerate(mat_d):
        mat_d[jj] = (np.asarray(mat_d[jj]))
    return names, mat_d


def run_pca(data):
    pca = PCA()
    pca_res = pca.fit_transform(np.transpose(data))
    variance = pca.explained_variance_ratio_
    feature_weights = pca.components_
    return pca_res, variance, feature_weights


def create_pc_feature_weights_df(names, feature_weights):
    pc1_features_df = pd.DataFrame(columns=names, index=[f'PC{i}' for i in list(range(len(names)))])
    for i, name in enumerate(names): 
        for j, pc in enumerate(pc1_features_df.index):
            pc1_features_df.loc[pc, name] = feature_weights[i, j]
    pc1_features_df['PC'] = pc1_features_df.index
    return pc1_features_df


def plot_pc1_feature_weights(feature_weights_df, names_cortex, filename):
    feature_weights_melt_df = feature_weights_df.melt(value_vars=names_cortex, id_vars='PC')
    
    plt.figure(figsize=(5,8))
    g = sns.barplot(y='variable', x='value', data=feature_weights_melt_df[feature_weights_melt_df.PC=='PC0'], palette='RdBu',
                   order=feature_weights_melt_df[feature_weights_melt_df.PC=='PC0'].sort_values(by='value', ascending=False)['variable'])
    g.set_yticklabels(g.get_yticklabels())
    g.set_ylabel("ENIGMA disease")
    g.set_xlabel("PC1 Feature Weight")
    plt.savefig(filename, transparent=True, dpi=300, bbox_inches='tight')
    