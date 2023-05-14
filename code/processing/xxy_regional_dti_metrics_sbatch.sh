#!/bin/bash

#SBATCH --array=1-256
#SBATCH --time=1:00:00

module load python/3.7
# extract subjects
SUBJECT=$( sed "${SLURM_ARRAY_TASK_ID}q;d" /data/DNU/liza/data/XXY/demo_data/XXY_3TC_PseudoGUIDs.txt )
DTI_METRICS_DIR=/data/DNU/liza/data/XXY/Nifti/derivatives/tractoflow
PARCELLATION_FILE=/data/DNU/liza/data/XXY/Nifti/derivatives/Freesurfer/sub-${SUBJECT}/parcellation/HCP.nii.gz
ROI_LABELS_FILE=/data/DNU/liza/code/NIH_Multimodal_SCA_Phenotypes/data/hcp_rois.txt

python compute_regional_dti_metrics.py --sub $SUBJECT --tractoflow_dir $DTI_METRICS_DIR --parcellation "HCP" --parcellation_file $PARCELLATION_FILE --roi_labels_file $ROI_LABELS_FILE
