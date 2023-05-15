opt.analysis_name = 'aneuploidyPC1reverse_celltypes';
opt.atlas = 'HCP';
opt.n_nulls = 1000;
opt.n_rois = 344;
load('../data/pc1_avg_withTBV_bcoef.mat');
opt.phenotype_data = -1 * pc1Avg;
opt.dir_result = 'gcea';
opt.GCEA.dataset = 'PsychEncode-cellTypesTPM-discrete';
opt.phenotype_nulls = '../data/pc1_avg_null_spins.mat';
opt.aba_mat = '../data/hcp_ahba_expression_344ROI.csv';

cTable_celltypes = ABAnnotate(opt);
