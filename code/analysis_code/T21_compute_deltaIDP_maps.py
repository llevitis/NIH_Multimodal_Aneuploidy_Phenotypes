"""
Compute T21 delta IDP maps
"""

import os
import json
import sys
import importlib
sys.path.insert(0, "../../code/utils/")
import utils
importlib.reload(utils)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import nibabel as nib 
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests
import collections

"""
Load data
"""

demo_data = pd.read_csv("../../data/DS/ds_demo_data.csv", index_col=0)
euler_data = pd.read_csv("../../data/DS/00_group2_stats_tables/euler.tsv", sep="\t")
euler_data['subject'] = [x.split("-")[1] for x in euler_data['subject']]

# Identify subjects who pass QC 
subs_mprage_rawquality_pass = demo_data[demo_data.mprage_raw_quality != "3_poor"].PseudoGUID
subs_euler_pass = euler_data[euler_data.mean_euler_bh > -217].subject
subs_qc_pass = list(set(subs_mprage_rawquality_pass) & set(subs_euler_pass))

# Get total tissue volume (TTV)

t21_aseg = pd.read_csv("../../data/DS/00_group2_stats_tables/aseg.tsv", sep="\t", index_col=0)
t21_aseg.index = [x.split("-")[1] for x in t21_aseg.index]

for sub in t21_aseg.index: 
    t21_aseg.loc[sub, 'TTV'] = t21_aseg.loc[sub, 'BrainSegVolNotVent']

# Update the HCP ROI names 

with open('../../data/hcp_rois.txt', 'r') as f: 
    hcp_rois = f.read().split("\n")[:-1]

hcp_rois_fs = []
for roi in hcp_rois:
    for roi2 in ['p9-46v', 'a9-46v', '9-46d', 'i6-8', 's6-8', 'OP2-3']: 
        if roi2 in roi:
            roi2_new = roi2.replace("-", "_")
            roi = roi.replace(roi2, roi2_new)
    hcp_rois_fs.append(roi)  

hcp_renaming_map = {}
for i, roi in enumerate(hcp_rois_fs):
    hcp_renaming_map[roi] = hcp_rois[i]

# Set the phenotypes 

morpho_idp = ['thickness', 'meancurv', 'volume', 'area', 'curvind', 'foldind', 'gauscurv']
dti_idp = ['fa', 'md', 'ad', 'rd', 'ga', 'mode']
func_idp = ['reho', 'alff']
mm_idp = morpho_idp + dti_idp + func_idp

# Create clean dictionary of subject by ROI dataframes for each IDP 
# separately for sMRI, dMRI, and rsFMRI modalities 

# sMRI
t21_morphometric_hcp_data = {} 
for i,idp in enumerate(morpho_idp):  
    t21_morphometric_hcp_data[idp] = pd.read_csv("../../data/DS/00_group2_stats_tables/PARC_HCP_" + idp + ".csv")
    pseudo_guids = list(t21_morphometric_hcp_data[idp][t21_morphometric_hcp_data[idp].columns[0]])
    t21_morphometric_hcp_data[idp].index = [x.split("-")[1] for x in pseudo_guids]
    # remove '_thickness' or the other morphometric feature from the column name to get common column names
    cols = [col.replace('_' + idp, '') for col in t21_morphometric_hcp_data[idp].columns]
    t21_morphometric_hcp_data[idp].columns = cols
    t21_morphometric_hcp_data[idp].rename(columns=hcp_renaming_map, inplace=True)


# dMRI
t21_dwi_hcp_data = {} 
for i,idp in enumerate(dti_idp):  
    t21_dwi_hcp_data[idp] = pd.read_csv(f"../../data/DS/dwi_parc-HCP_{idp}.csv", index_col=0)
    t21_dwi_hcp_data[idp].rename(columns=hcp_renaming_map, inplace=True)
subs_dwi_qc_pass = list(set(subs_qc_pass) & set(t21_dwi_hcp_data['fa'].index))

# rsFMRI 

reho_glasser_df = pd.read_csv("../../data/DS/reho_glasser_func_rois_to_keep.csv", index_col=0)
alff_glasser_df = pd.read_csv("../../data/DS/alff_glasser_func_rois_to_keep.csv", index_col=0)

func_rois_to_use = reho_glasser_df.columns[0:344]

with open('../../data/DS/fmri_subs_qc_pass.txt', 'r') as f:
    subs_fmri_qc_pass = f.read().split('\n')[:-1] 

t21_func_hcp_data = {}
t21_func_hcp_data['reho'] = reho_glasser_df
t21_func_hcp_data['alff'] = alff_glasser_df
for idp in func_idp: 
    t21_func_hcp_data[idp].rename(columns=hcp_renaming_map, inplace=True)

# Add demographic data to the modality specific dictionaries 
subs_qc_pass_mm = [subs_qc_pass, subs_dwi_qc_pass, subs_fmri_qc_pass]
mm_idp_list = [morpho_idp, dti_idp, func_idp]
for i, modality_dict in enumerate([t21_morphometric_hcp_data, t21_dwi_hcp_data, t21_func_hcp_data]):
    for idp in mm_idp_list[i]:
        for sub in subs_qc_pass_mm[i]:
            modality_dict[idp].loc[:, 'TTV'] = t21_aseg.loc[:, 'TTV']
            modality_dict[idp].loc[sub, 'dx_group'] = demo_data[demo_data.PseudoGUID == sub].GROUP.values[0]
            modality_dict[idp].loc[sub, 'sex'] = demo_data[demo_data.PseudoGUID == sub].SEX.values[0]
            modality_dict[idp].loc[sub, 'age'] = demo_data[demo_data.PseudoGUID == sub].SCANAGE.values[0]
            modality_dict[idp].loc[sub, 'euler_mean_bh'] = euler_data[euler_data.subject == sub].mean_euler_bh.values[0]
            if modality_dict[idp].loc[sub, 'dx_group'] == 'HV': 
                modality_dict[idp].loc[sub, 'SCdose'] = 0
            elif modality_dict[idp].loc[sub, 'dx_group'] == 'DS': 
                modality_dict[idp].loc[sub, 'SCdose'] = 1

# Save the IDP specific dataframes

for i, idp in enumerate(mm_idp):
    if idp in morpho_idp:
        idp_df = t21_morphometric_hcp_data[idp].loc[subs_qc_pass,:]
    elif idp in dti_idp:
        idp_df = t21_dwi_hcp_data[idp].loc[subs_dwi_qc_pass,:]
    elif idp in func_idp:
        idp_df = t21_func_hcp_data[idp].loc[subs_fmri_qc_pass,:]
    idp_df.to_csv(f"../../data/t21_{idp}.csv")


"""
Compute the effect sizes using TTV corrected data 
"""

scd_effect_idp_withTTV_dict = {}

# sMRI 
for i, idp in enumerate(morpho_idp):
    idp_df = t21_morphometric_hcp_data[idp]
    scd_effect_idp_withTTV_dict[idp] = utils.scd_effect_on_mf(idp_df.loc[subs_qc_pass,:], 
                                                              hcp_rois, 
                                                              covar_args=['euler_mean_bh', 'sex', 'TTV'])
# dMRI 
for i, idp in enumerate(dti_idp):
    idp_df = t21_dwi_hcp_data[idp]
    scd_effect_idp_withTTV_dict[idp] = utils.scd_effect_on_mf(idp_df.loc[idp_df.index.intersection(subs_dwi_qc_pass)], 
                                                              hcp_rois, 
                                                              covar_args=['euler_mean_bh', 'sex', 'TTV'])

# rsfMRI 
for i, idp in enumerate(func_idp):
    idp_df = t21_func_hcp_data[idp]
    scd_effect_idp_withTTV_dict[idp] = utils.scd_effect_on_mf(idp_df.loc[subs_fmri_qc_pass,:], 
                                                          func_rois_to_use, 
                                                          covar_args=['euler_mean_bh', 'sex', 'TTV', 'FD'])


"""
Compute the effect sizes using TTV un-corrected data 
"""

scd_effect_idp_withoutTTV_dict = {}

# sMRI 
for i, idp in enumerate(morpho_idp):
    idp_df = t21_morphometric_hcp_data[idp]
    scd_effect_idp_withoutTTV_dict[idp] = utils.scd_effect_on_mf(idp_df.loc[subs_qc_pass,:], 
                                                             hcp_rois, 
                                                             covar_args=['euler_mean_bh', 'sex'])
# dMRI 
for i, idp in enumerate(dti_idp):
    idp_df = t21_dwi_hcp_data[idp]
    scd_effect_idp_withoutTTV_dict[idp] = utils.scd_effect_on_mf(idp_df.loc[idp_df.index.intersection(subs_dwi_qc_pass)], 
                                                              hcp_rois, 
                                                              covar_args=['euler_mean_bh', 'sex'])

# rsfMRI 
for i, idp in enumerate(func_idp):
    idp_df = t21_func_hcp_data[idp]
    scd_effect_idp_withoutTTV_dict[idp] = utils.scd_effect_on_mf(idp_df.loc[subs_fmri_qc_pass,:], 
                                                          func_rois_to_use, 
                                                          covar_args=['euler_mean_bh', 'sex', 'FD'])


# Construct a dataframe with the t-statistics, b-coefficients, and FDR corrected q-values 
# Constrained to the ROIs with coverage for rsFMRI

scd_effect_withTTV_df = pd.DataFrame(index=func_rois_to_use)
scd_effect_withoutTTV_df = pd.DataFrame(index=func_rois_to_use)

for idp in mm_idp: 
    scd_effect_withTTV_df.loc[func_rois_to_use, idp] = scd_effect_idp_withTTV_dict[idp]['SCdose_t']
    scd_effect_withTTV_df.loc[func_rois_to_use, f'{idp}_FDR'] = scd_effect_idp_withTTV_dict[idp]['SCdose_FDR']
    scd_effect_withTTV_df.loc[func_rois_to_use, f'{idp}_bcoef'] = scd_effect_idp_withTTV_dict[idp]['SCdose_bcoef']
    scd_effect_withoutTTV_df.loc[func_rois_to_use, idp] = scd_effect_idp_withoutTTV_dict[idp]['SCdose_t']
    scd_effect_withoutTTV_df.loc[func_rois_to_use, f'{idp}_FDR'] = scd_effect_idp_withoutTTV_dict[idp]['SCdose_FDR']
    scd_effect_withoutTTV_df.loc[func_rois_to_use, f'{idp}_bcoef'] = scd_effect_idp_withoutTTV_dict[idp]['SCdose_bcoef']

scd_effect_withTTV_df.to_csv("../../data/t21_scd_bcoef_effect_withTTV.csv")
scd_effect_withoutTTV_df.to_csv("../../data/t21_scd_bcoef_effect_withoutTTV.csv")