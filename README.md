The variegation of human brain vulnerability to rare genetic disorders and convergence with behaviorally defined disorders

### Description
This repository contains all the code used for analyses in the following preprint: https://www.biorxiv.org/content/10.1101/2022.11.12.516252v1.abstract. The scripts in `code/analysis_code` are split up thematically and rely on data that is currently stored in a private Git repo. General utility functions put together specifically for these analyses are stored in `code/utils` and are called in the `code/analysis_code` scripts. 

Here is a breakdown of the scripts (in the order in which they're meant to be run, as some scripts rely on data generated in preceding scripts).

#### Computing regional imaging derived phenotype (IDP) changes in each aneuploidy 

Each of the following scripts loads subject by brain region matrices for all the 15 IDPs of interest and generates IDP-specific matrices with regional effect sizes and FDR corrected q-values. This is done both with and without total tissue volume correction. 
*  `code/analysis_code/XXY_compute_deltaIDP_maps.py`
* `code/analysis_code/XYY_compute_deltaIDP_maps.py`
* `code/analysis_code/T21_compute_deltaIDP_maps.py`