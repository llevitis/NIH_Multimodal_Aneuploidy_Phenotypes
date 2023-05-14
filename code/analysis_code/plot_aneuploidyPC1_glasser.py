import importlib
import os
import json
import sys

from glob import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

import seaborn as sns

from scipy import stats

import nibabel as nib 

import utils

from netneurotools.plotting import plot_fsaverage
from pathlib import Path

scd_bcoef_withTTV_allDisorders_full = pd.read_csv("../../data/scd_bcoef_allDisorders_subsAllMods_df.csv", index_col=0)

with open('../../data/hcp_rois.txt', 'r') as f: 
    hcp_rois_orig = f.read().split("\n")[:-1]
    
# need to change some '_' to '-' to match with proper HCP naming convention
hcp_rois = []
for roi in hcp_rois_orig:
    for roi2 in ['p9_46v', 'a9_46v', '9_46d', 'i6_8', 's6_8', 'OP2_3']: 
        if roi2 in roi:
            roi2_new = roi2.replace("_", "-")
            roi = roi.replace(roi2, roi2_new)
    hcp_rois.append(roi)


opts = dict(
    views=['lat', 'med'], colormap='RdBu_r', vmin=-.5, vmax=.5
)

#scd_bcoef_withTBV_allDisorders_df = pd.read_csv("../../data/scd_bcoef_withTBV_allDisorders_df.csv", index_col=0)
func_rois_to_use = scd_bcoef_withTTV_allDisorders_full.index


hcp_lh_annot = '/Users/levitise2/code/NIH_Multimodal_SCA_Phenotypes/data/lh.HCPMMP1.annot'
hcp_rh_annot = '/Users/levitise2/code/NIH_Multimodal_SCA_Phenotypes/data/rh.HCPMMP1.annot'


# curr_data = []
# for i, roi in enumerate(hcp_rois):
#     if roi in func_rois_to_use: 
#         curr_data.append(scd_bcoef_withTBV_allDisorders_full.loc[roi, 'avg'])
#     else: 
#         curr_data.append(np.nan)
# brain = plot_fsaverage(curr_data,
#                         lhannot=hcp_lh_annot, rhannot=hcp_rh_annot,
#                         order='lr',
#                         data_kws={'representation': "wireframe"}, noplot='???',
#                         colorbar=False,
#                         **opts)
# brain.save_image(f'../../figures/brainplots/aneuploidy_PC1_avg_withTBV_parc-HCP.png')

for dataset in ['xxy', 'xyy', 't21']:
    curr_data = []
    for i, roi in enumerate(hcp_rois):
        if roi in func_rois_to_use: 
            curr_data.append(scd_bcoef_withTTV_allDisorders_full.loc[roi, dataset])
        else: 
            curr_data.append(np.nan)
    brain = plot_fsaverage(curr_data,
                            lhannot=hcp_lh_annot, rhannot=hcp_rh_annot,
                            order='lr',
                            data_kws={'representation': "wireframe"}, noplot='???',
                            colorbar=False,
                            **opts)
    brain.save_image(f'../../figures/brainplots/{dataset}_PC1_subsAllMods_parc-HCP.png')