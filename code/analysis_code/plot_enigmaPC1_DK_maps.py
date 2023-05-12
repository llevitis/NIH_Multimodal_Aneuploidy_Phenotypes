import pandas as pd
from netneurotools.plotting import plot_fsaverage 
from enigmatoolbox.datasets.base import load_summary_stats
from netneurotools import datasets
import nibabel as nib
import numpy as np


scale = "scale033"
cammoun = datasets.fetch_cammoun2012()
info = pd.read_csv(cammoun['info'])
cortex = info.query('scale == @scale & structure == "cortex"')['id']
cortex = np.array(cortex) - 1  # python indexing
nnodes = len(cortex)
annot = datasets.fetch_cammoun2012('fsaverage')[scale]

phenotype = 'CortSurfCortThick'
enigma_dkt_pc1_df = pd.read_csv(f"../../data/enigma_dkt_pc1_{phenotype}_df.csv", index_col='Structure')

## convert order to cammoun order
# enigma structure names
#print(enigma_dkt_pc1_df.index)
# Load summary statistics for ENIGMA-22q
sum_stats = load_summary_stats('22q')
enigma_structure = sum_stats['CortThick_case_vs_controls']['Structure'] 
for i in range(len(enigma_structure)):
    _, enigma_structure[i] = enigma_structure[i].split('_')

# cammoun structure names
cammoun_structure = info.query('scale == "scale033"')['label']
cammoun_structure = cammoun_structure[cortex]

# match them up
left_cammoun = cammoun_structure[34:]
left_enigma = enigma_structure[:34]
left_reorder = pd.Series(left_enigma.index, index=left_enigma). \
                                                reindex(left_cammoun)
right_cammoun = cammoun_structure[:34]
right_enigma = enigma_structure[34:]
right_reorder = pd.Series(right_enigma.index, index=right_enigma). \
                                                reindex(right_cammoun)

# indeces required to reorder enigma data to cammoun order
reorder = np.concatenate((right_reorder.to_numpy(), left_reorder.to_numpy()))  
print(reorder)
opts = dict(
            views=['lat', 'med'],                                                           
            colormap='RdBu_r', colorbar=False, vmin=-0.5, vmax=0.5
           )

for plot_type in ['AneuploidyPC1', 'BehavDisordersPC1', 'BehavNEDisordersPC1']: 
    # curr_data = []                                                                      
    noplot=['???'] 
    curr_data = list(enigma_dkt_pc1_df[plot_type])
    # reorder using the enigma to cammoun mapping
    curr_data = [curr_data[i] for i in reorder]
    brain = plot_fsaverage(curr_data,
                           lhannot=annot.lh, rhannot=annot.rh,
                           order='rl',
                           subject_id='fsaverage', data_kws={'representation': "wireframe"}, noplot=noplot,
                           **opts)
    brain.save_image(f'../../figures/brainplots/{plot_type}_{phenotype}.png')


