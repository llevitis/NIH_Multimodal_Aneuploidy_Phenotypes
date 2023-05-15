### Description
This repository contains all the code used for analyses in the following preprint: **"The variegation of human brain vulnerability to rare genetic disorders and convergence with behaviorally defined disorders"** ([preprint](https://www.biorxiv.org/content/10.1101/2022.11.12.516252v1.abstract)). The scripts in `code/analysis_code` are split up thematically and rely on data that is currently stored in a private Git repo. General utility functions put together specifically for these analyses are stored in `code/utils` and are called in the `code/analysis_code` scripts. 

Here is a breakdown of the scripts (in the order in which they're meant to be run, as some scripts rely on data generated in preceding scripts).

#### Computing regional imaging derived phenotype (IDP) changes in each aneuploidy 

Each of the following scripts loads subject by brain region matrices for all the 15 IDPs of interest and generates IDP-specific matrices with regional effect sizes and FDR corrected q-values. This is done both with and without total tissue volume correction. 
*  `code/analysis_code/XXY_compute_deltaIDP_maps.py`
* `code/analysis_code/XYY_compute_deltaIDP_maps.py`
* `code/analysis_code/T21_compute_deltaIDP_maps.py`

Together with the plotting functionality in `code/scripts/plot_aneuploidy_idp_effect_size_maps.py`, these scripts generate the brain maps seen in **Figure 2a**. 

#### Defining organizing-principles of cortical change across IDPs, regions and aneuploidies  

The followings script aggregates the aneuploidy-specific brain region by IDP matrices to evaluate how the brain maps relate across phenotypes and aneuploidies. The hierarchical clustering plot and force-directed edge graph (**Fig 2b, c**) are generated using this script. 

* `code/analysis_code/Hierarchical_Clustering.py`


#### Defining and characterizing principal components of multimodal cortical change in each aneuploidy

The following script performs principal component analysis (PCA) on each of the aneuploidy-specific brain region by delta IDP matrices. It compares the feature loadings and principal component scores for the first principal component of each aforementioned matrix (**Fig 3**). Additionally, this script averages the PC1 maps across the aneuploidies to enable identification of a shared spatial axis of multimodal change (**Fig 4a**). Finally, this script contains code to carry out biological annotation of the PC1 maps using a previously defined annotation (**Fig 4c**)

* `code/analysis_code/Principal_Component_Analysis.py` 

#### Convergent multimodal cortical changes across aneuploidies and links to convergence across behaviorally-defined  disorders 

The following script leverages the [ENIGMA toolbox](https://enigma-toolbox.readthedocs.io/en/latest/) to download morphometric case-control effect size maps across behaviorally defined psychiatric disorders. PCA is then applied to derive a principal cross-disorder morphometric map, and this is compared with the average cross-aneuploidy multimodal map derived in the preceding script. 

* `code/analysis_code/ENIGMA_Aneuploidy_Comparison.py`

#### Gene-category enrichment analysis of the cross-disorder aneuploidy map 

To gain a better understanding of how our average cross-aneuploidy multimodal map may relate to biological processes and cell types, we use several previously published tools - namely, [abagen](https://abagen.readthedocs.io/en/stable/) and [ABAnnotate](https://zenodo.org/record/6463329).  

The following script calls `abagen` to download the Allen Human Brain Atlas and create a gene expression matrix based on the multimodal HCP parcellation. 

* `code/abagen_compute_expression_matrix.py` 

We then use the `ABAnnotate` toolbox to perform ensemble gene-category enrichment analysis (GCEA) using two different gene category annotations.  

* GO Biological Processes: `code/gcea/AneuploidyPC1_analysis_celltypes.m`
* PsychEncode neuronal cell type markers: `code/gcea/AneuploidyPC1_analysis_celltypes.m`