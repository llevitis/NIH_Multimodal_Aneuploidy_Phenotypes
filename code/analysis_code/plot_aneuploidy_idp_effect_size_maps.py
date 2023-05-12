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

scd_bcoef_effect_files = glob('../../data/*scd_bcoef_effect_*.csv')

scd_bcoef_effect_dict = {} 
for file in scd_bcoef_effect_files:
    key = file.split("/")[-1].split("_")[0] + '_' + file.split("/")[-1].split("_")[4].split(".")[0]
    scd_bcoef_effect_dict[key] = pd.read_csv(file, index_col=0)


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

func_rois_to_use = list(scd_bcoef_effect_dict['xxy_withoutTBV'].index)

bcoef=True

t1w_dti_func_features = ['thickness', 'meancurv', 'volume', 'area', 'curvind', 'foldind', 
                            'gauscurv', 'fa', 'md', 'ad', 'rd', 'ga', 'mode', 
                            'reho', 'alff']
if bcoef==True:
    t1w_dti_func_features = [x + '_bcoef' for x in t1w_dti_func_features]

hcp_lh_annot = '/Users/levitise2/code/NIH_Multimodal_SCA_Phenotypes/data/lh.HCPMMP1.annot'
hcp_rh_annot = '/Users/levitise2/code/NIH_Multimodal_SCA_Phenotypes/data/rh.HCPMMP1.annot'

if bcoef==False:
    vmin=-5
    vmax=5
else:
    vmin=-1
    vmax=1
opts = dict(
    views=['lat', 'med'], colormap='RdBu_r', vmin=vmin, vmax=vmax
)

for key in ['xxy_withTBV', 'xyy_withTBV', 'ds_withTBV']:
    aneuploidy = key.split("_")[0]
    tbv_status = key.split("_")[1]
    curr_df = scd_bcoef_effect_dict[key]
    for idp in t1w_dti_func_features: 
        curr_data = []
        for i, roi in enumerate(hcp_rois):
            if roi in func_rois_to_use: 
                curr_data.append(curr_df.loc[roi, idp])
            else: 
                curr_data.append(np.nan)
        brain = plot_fsaverage(curr_data,
                               lhannot=hcp_lh_annot, rhannot=hcp_rh_annot,
                               order='lr',
                               data_kws={'representation': "wireframe"}, noplot='???',
                               colorbar=False,
                               **opts)
        if bcoef==False:
            brain.save_image(f'../../figures/brainplots/hcp_{aneuploidy}_{idp}_{tbv_status}_tval.png')
        else:
            brain.save_image(f'../../figures/brainplots/hcp_{aneuploidy}_{idp}_{tbv_status}_bcoef.png')