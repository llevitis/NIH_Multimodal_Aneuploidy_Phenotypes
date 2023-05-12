import abagen
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

files = abagen.fetch_microarray(donors='all', verbose=0)
data = files['9861']
probes = abagen.io.read_probes(data['probes'])

with open('../../data/hcp_rois.txt', 'r') as f: 
    hcp_rois = f.read().split("\n")[:-1]

atlas = {}
atlas['image'] = "../../data/HCP_renumbered_int.nii.gz"
atlas['info'] = "../../data/HCP_renumbered_int.csv"

expression = abagen.get_expression_data(atlas['image'], atlas['info'],
                                        missing='interpolate', norm_matched=False)

expression.to_csv("../../data/hcp_ahba_expression_344ROI.csv")

