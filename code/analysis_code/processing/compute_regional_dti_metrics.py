import nibabel as nib
import nilearn.plotting as plotting
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

from scipy.io import savemat, loadmat

import pandas as pd
import seaborn as sns

import glob
import os
from argparse import ArgumentParser

def renum_aparc500_atlas(aparc500_atlas): 
    aparc500_data = aparc500_atlas.get_fdata()

    # set all the values that aren't in the proper HCP ROIs to 0 and renumber the remaining ones
    aparc500_data[aparc500_data < 1001] = 0
    idx = 0
    for val in sorted(list(set(aparc500_data.flatten()))): 
        aparc500_data[aparc500_data == val] = idx 
        idx += 1

    aparc500_atlas_renum = nib.Nifti1Image(aparc500_data, aparc500_atlas.affine, aparc500_atlas.header)
    return aparc500_atlas_renum

def renum_hcp_atlas(hcp_atlas):
    hcp_data = hcp_atlas.get_fdata()

    # set all the values that aren't in the proper HCP ROIs to 0 and renumber the remaining ones
    hcp_data[hcp_data < 1001] = 0
    idx = 0
    for val in sorted(list(set(hcp_data.flatten()))): 
        hcp_data[hcp_data == val] = idx 
        idx += 1

    hcp_atlas_renum = nib.Nifti1Image(hcp_data, hcp_atlas.affine, hcp_atlas.header)
    return hcp_atlas_renum 


def main():
    parser = ArgumentParser()
    parser.add_argument("--sub",
                        help="PseudoGUID of interest.")
    parser.add_argument("--tractoflow_dir",
                        help="Please provide path to the TractoFlow output directory")
    parser.add_argument("--parcellation", 
                        help="Name of parcellation")
    parser.add_argument("--parcellation_file",
                        help="Please provide path to the parcellation file of interest")
    parser.add_argument("--roi_labels_file", 
                        help="Please provide path to ROI labels file")                    
    results = parser.parse_args()

    sub = results.sub
    tractoflow_dir = results.tractoflow_dir
    parcellation = results.parcellation
    parcellation_file = results.parcellation_file
    roi_labels_file = results.roi_labels_file

    parcellation_img = nib.load(parcellation_file)
    if parcellation == 'HCP':
        parcellation_img = renum_hcp_atlas(parcellation_img)
    elif parcellation == 'aparc500': 
        parcellation_img = renum_aparc500_atlas(parcellation_img)

    with open(roi_labels_file, 'r') as f:
        roi_labels = f.read().split("\n")[:-1]

    metrics = ['fa', 'md', 'ad', 'rd', 'ga', 'mode']
    for metric in metrics:
        metric_file = glob.glob(os.path.join(tractoflow_dir, "sub-{0}_ses-v01".format(sub), "DTI_Metrics", "*{0}.nii.gz".format(metric)))[0]
        metric_img = nib.load(metric_file)
        metric_img = resample_to_img(metric_img, parcellation_img)
        masker = NiftiLabelsMasker(labels_img=parcellation_img, standardize=True,
                           memory='nilearn_cache', verbose=5)
        metric_arr = masker.fit_transform([metric_img])[0]
        metric_mat = pd.DataFrame(index = [sub], columns = roi_labels)
        metric_mat.loc[sub, roi_labels] = metric_arr
        metric_mat.to_csv(os.path.join(tractoflow_dir, "sub-{0}_ses-v01".format(sub), "DTI_Metrics", 
                                        "sub-{0}_ses-v01_parcellation-{1}_{2}.csv".format(sub, parcellation, metric)))


if __name__ == "__main__":
    main()
