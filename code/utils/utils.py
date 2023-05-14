import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.preprocessing import MinMaxScaler
from PIL import ImageColor
import nibabel as nib 
import nilearn.plotting as plotting
from nilearn import datasets
from nilearn import surface

def euler_effect_on_mf(mf_df, roi_ids): 
    euler_effect_mf_df = pd.DataFrame(index=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ C(SEX) + age + euler_mean_bh".format(roi),
                            data=mf_df).fit()
        t = fitmod.tvalues[3]
        p = fitmod.pvalues[3]
        euler_effect_mf_df.loc[roi,'euler_p'] = p
        euler_effect_mf_df.loc[roi,'euler_t'] = t
    
    fdrs = multipletests(euler_effect_mf_df['euler_p'].values,method='fdr_bh')
    euler_effect_mf_df.loc[:,'euler_FDR'] = fdrs[1]
    
    return euler_effect_mf_df


# function to calculate Cohen's d for independent samples
def cohend(d1, d2, sd_type):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	if sd_type=='pooled':
	    return (u1 - u2) / s
	elif sd_type=='control': 
	    return (u1 - u2) / s2

def scd_effect_on_mf_cohen_d(mf_df, roi_ids, covar_args):
    '''
    Given a sub x roi dataframe of morphometric features and several demographic variables, 
    this function computes the standardized mean differences (Cohen's d) between cases and controls 
    across ROIs after regressing out the covariates. 
    
    mf_df -- a sub x roi dataframe
    roi_ids -- a list of labels corresponding to the ROI columns in mf_df
    '''
    scd_effect_mf_df = pd.DataFrame(index=roi_ids, columns=['Cohen_d'])
    resid_df = pd.DataFrame(index=mf_df.index, columns=roi_ids)
    ols_eq = "Q('{0}') ~ age + sex"
    for covar in covar_args:
        ols_eq += f' + {covar}'
    for roi in roi_ids:
        fitmod = smf.ols(ols_eq.format(roi), 
                         data=mf_df, 
                         missing='drop').fit()
        resid_df.loc[:,roi] = fitmod.resid
    resid_df['SCdose'] = mf_df['SCdose']
    for roi in roi_ids:
        roi_cohend = cohend(resid_df[resid_df.SCdose==1][roi], resid_df[resid_df.SCdose==0][roi])
        scd_effect_mf_df.loc[roi, 'Cohen_d'] = roi_cohend
    return scd_effect_mf_df


def scd_effect_on_mf(mf_df, roi_ids, covar_args):
    '''
    Given a sub x roi dataframe of morphometric features and several demographic variables, 
    this function first standardizes data within a given IDP across ROIs and
    computes the effect of sex chromosome dosage on the morphometric 
    feature of interest at each ROI. Additionally, correction for multiple comparison
    is done, and a dataframe with t & p values for each ROI is returned.
    
    mf_df -- a sub x roi dataframe
    roi_ids -- a list of labels corresponding to the ROI columns in mf_df
    '''
    scd_effect_mf_df = pd.DataFrame(index=roi_ids)
    ols_eq = "Q('{0}') ~ age + SCdose"
    for covar in covar_args:
        ols_eq += f' + {covar}'
    for roi in roi_ids:
        roi_data = mf_df[roi]
        roi_mean = roi_data.mean()
        roi_std = roi_data.std(ddof=0)
        if roi_std == 0:
            roi_std = 0.0000001  # Assign a small value if std is zero
        roi_zscore_data = pd.DataFrame(index=mf_df.index, columns=[f'{roi}_zscore'])
        roi_zscore_data[f'{roi}_zscore'] = (roi_data - roi_mean) / roi_std
        mf_df = pd.concat([mf_df, roi_zscore_data], axis=1)
        roi_zscore = roi + '_zscore'
        fitmod = smf.ols(ols_eq.format(roi_zscore), data=mf_df, missing='drop').fit()
        bcoef = fitmod.params['SCdose']
        t = fitmod.tvalues['SCdose']
        p = fitmod.pvalues['SCdose']
        scd_effect_mf_df.loc[roi, 'SCdose_p'] = p
        scd_effect_mf_df.loc[roi, 'SCdose_t'] = t
        scd_effect_mf_df.loc[roi, 'SCdose_bcoef'] = bcoef
    
    fdrs = multipletests(scd_effect_mf_df['SCdose_p'].values,method='fdr_bh')
    scd_effect_mf_df.loc[:,'SCdose_FDR'] = fdrs[1]
    scd_effect_mf_df.loc[:,'SCdose_reject'] = fdrs[0]
    
    return scd_effect_mf_df
    
def compute_mf_norm_corr_matrix(mf_df, roi_ids):
    '''
    Given a sub x roi dataframe of morphometric features and several demographic variables, 
    this function first regresses out the effect of age, sex, and age*sex on the feature of interest. 
    It subsequently computes a correlation matrix for that feature only in individuals
    with a SC dosage of 2 (either XX or XY).
    
    mf_df -- a sub x roi dataframe
    roi_ids -- a list of labels corresponding to the ROI columns in mf_df
    '''
    mf_norm_sex_age_residuals = pd.DataFrame(index=mf_df[mf_df.SCdose==0].index, columns=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ C(SEX) + age + C(SEX) * age".format(roi),
                         data=mf_df[mf_df.SCdose==0]).fit()
        mf_norm_sex_age_residuals.loc[:,roi] = fitmod.resid
        
    mf_norm_corr_matrix = mf_norm_sex_age_residuals.corr() # rmap_a
    return mf_norm_corr_matrix

def compute_rmap_b_mf(mf_norm_corr_matrix, scd_effect_mf_df, roi_ids):
    r_map_b_mf = []
    for i, roi in enumerate(roi_ids): 
        r,p = stats.pearsonr(mf_norm_corr_matrix[roi], scd_effect_mf_df['SCdose_t'])
        r_map_b_mf.append(r)
    return r_map_b_mf

def compute_scd_effect_mf_resid(mf_df, roi_ids, include_euler=True, include_TTV=False):
    resid_df = pd.DataFrame(index=mf_df.index, columns=roi_ids)
    for roi in roi_ids:
        if include_TTV == False:
            fitmod = smf.ols("Q('{0}') ~ age + SCdose + euler_mean_bh".format(roi),
                            data=mf_df).fit()
        elif include_TTV == True:
            fitmod = smf.ols("Q('{0}') ~ age + SCdose + euler_mean_bh + TTV".format(roi),
                            data=mf_df).fit()
        resid_df.loc[:,roi] = fitmod.resid
    return resid_df

def compute_mf_resid(mf_df, roi_ids, include_TTV=False, include_FD=False):

    resid_df = pd.DataFrame(index=mf_df.index, columns=roi_ids)
    for roi in roi_ids:
        if include_TTV == False:
            fitmod = smf.ols("Q('{0}') ~ age + euler_mean_bh".format(roi),
                            data=mf_df,
                            missing='drop').fit()
        elif include_TTV == True:
            fitmod = smf.ols("Q('{0}') ~ age + euler_mean_bh + TTV".format(roi),
                            data=mf_df,
                            missing='drop').fit()
        elif include_FD == True and include_TTV == False:
            fitmod = smf.ols("Q('{0}') ~ age + euler_mean_bh + FD".format(roi),
                            data=mf_df,
                            missing='drop').fit()
        elif include_FD == True and include_TTV == True:
            fitmod = smf.ols("Q('{0}') ~ age + euler_mean_bh + TTV + FD".format(roi),
                            data=mf_df,
                            missing='drop').fit()
        resid_df.loc[:,roi] = fitmod.resid
    resid_df['SCdose'] = mf_df['SCdose']
    return resid_df

def compute_msn(mf_df_dict, subs, features, roi_ids):
    msn_df = pd.DataFrame(index=subs, columns=roi_ids)
    for sub in subs:
        sub_mf_df = pd.DataFrame(index=roi_ids, columns=features)
        for feature in features: 
            sub_mf_df.loc[roi_ids, feature] = mf_df_dict[feature].loc[sub, roi_ids]
        sub_mf_df = sub_mf_df.astype(float)
        sub_mf_zscore_df = sub_mf_df.apply(stats.zscore)
        sub_msn_df = sub_mf_zscore_df.transpose().corr()
        np.fill_diagonal(sub_msn_df.values,0)
        msn_df.loc[sub, roi_ids] = sub_msn_df.mean()
        msn_df = msn_df.astype(float)
    return msn_df

def compute_scd_effect_global_mf_coupling(mf_df, roi_ids, include_euler=True):
    '''
    Given a sub x roi dataframe of morphometric features and several demographic 
    variables, this function first regresses out the effect of SC dosage, age, 
    and age*SC dosage on the morphometric feature of interest. For each subject, 
    the average cortical value for the morphometric feature is returned. 
    It then computes the effect of SC dosage on global anatomical coupling and
    corrects for multiple comparison. A dataframe with t & p values for each ROI 
    is returned.
    
    mf_df -- a sub x roi dataframe
    roi_ids -- a list of labels corresponding to the ROI columns in mf_df
    '''
    mf_norm_age_scd_resid_df = pd.DataFrame(index=mf_df.index, columns=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ age + SCdose + euler_mean_bh + age * SCdose".format(roi),
                         data=mf_df).fit()
        mf_norm_age_scd_resid_df.loc[:,roi] = fitmod.resid
        
    for var in ['age', 'SCdose']: 
        mf_norm_age_scd_resid_df[var] = mf_df[var]
        
    for sub in mf_norm_age_scd_resid_df.index: 
        avg_mf = np.mean(mf_norm_age_scd_resid_df.loc[sub, roi_ids])
        mf_norm_age_scd_resid_df.loc[sub, 'Avg_MF'] = avg_mf
        
    # examine SCD effects on the global anatomical coupling of each cortical region
    scd_global_anat_coupling_df = pd.DataFrame(index=roi_ids)
    for roi in roi_ids:
        fitmod = smf.ols("Q('{0}') ~ Avg_MF + SCdose + Avg_MF * SCdose".format(roi),
                         data=mf_norm_age_scd_resid_df).fit()
        t = fitmod.tvalues[3]
        p = fitmod.pvalues[3]
        scd_global_anat_coupling_df.loc[roi,'SCdose_global_p'] = p
        scd_global_anat_coupling_df.loc[roi,'SCdose_global_t'] = t
    
    fdrs = multipletests(scd_global_anat_coupling_df['SCdose_global_p'].values,method='fdr_bh')
    scd_global_anat_coupling_df.loc[:,'SCdose_global_FDR'] = fdrs[1]

    return mf_norm_age_scd_resid_df, scd_global_anat_coupling_df
    
def compute_scd_mf_coupling_effect(mf_norm_age_scd_resid_df, roi_ids):
    scd_mf_coupling_effect_df = pd.DataFrame(index=roi_ids, columns=roi_ids)
    for i, roi in enumerate(roi_ids): 
        for j, roi2 in enumerate(roi_ids): 
            fitmod = smf.ols("Q('{0}') ~ Q('{1}') + SCdose + Q('{1}') * SCdose".format(roi, roi2),
                         data=mf_norm_age_scd_resid_df).fit()
            t = fitmod.tvalues[3]
            p = fitmod.pvalues[3]
            scd_mf_coupling_effect_df.loc[roi, roi2] = t
            
    scd_mf_coupling_effect_sym_df = pd.DataFrame(index=roi_ids, columns=roi_ids, dtype="float32")
    for i, roi in enumerate(roi_ids): 
        for j, roi2 in enumerate(roi_ids): 
            val1 = scd_mf_coupling_effect_df.loc[roi, roi2]
            val2 = scd_mf_coupling_effect_df.loc[roi2, roi]
            avg = np.mean([val1, val2])
            scd_mf_coupling_effect_sym_df.loc[roi, roi2] = avg
            scd_mf_coupling_effect_sym_df.loc[roi2, roi] = avg
            
    return scd_mf_coupling_effect_sym_df
    
    
def compute_r_map_c_mf(scd_mf_coupling_effect_sym_df, scd_effect_mf_df, roi_ids):
    r_map_c_mf = []
    for i, roi in enumerate(roi_ids):
        r,p = stats.pearsonr(scd_mf_coupling_effect_sym_df[roi], scd_effect_mf_df['SCdose_t'])
        r_map_c_mf.append(r)
    return r_map_c_mf
    
def create_surf_stat_img(stat_data, parc308_img):
        
    stat_img_data = np.zeros_like(parc308_img.get_fdata())
    for i,roi in enumerate(roi_ids): 
        stat_img_data[parc308_data==(i+41)] = stat_data[i]
    
    stat_img = nib.Nifti1Image(stat_img_data, 
                               affine=parc308_img.affine, 
                               header=parc308_img.header)
    return stat_img

    
def plot_corr_matrix(corr_matrix_df, xlabel="308 Regions", ylabel="308 Regions", title="Coupling Change Matrix", cmap="RdBu_r"):
    plt.figure(figsize=(6,5))
    sns.heatmap(np.array(corr_matrix_df), cmap=cmap)
    plt.title(title, fontsize=14)
    plt.xticks([])
    plt.xlabel(xlabel, fontsize=14)
    plt.yticks([])
    plt.ylabel(ylabel, fontsize=14)

def create_scd_effect_images_dict(scd_effect_dict, features, rois, parc_data, parc_img):
    '''
    This function requires the effect of SCdose to be computed for a sub by ROI matrix corresponding 
    to some feature (e.g. cortical thickness) of interest. For each feature the function 
    creates a Nifti1 image where all the statistically significant ROIs and their corresponding t-values
    are projected. This is saved as a dictionary of images indexed by feature name.
    
    scd_effect_dict -- a dictionary containing ROI by stat measure (t-val, FDR-val) matrices, 
    each reflecting the effect of SCD on a feature of interest
    features -- a list of all features included in scd_effect_dict
    rois - a list of all ROIs
    parc_data - Nifti1 image data for the parcellation scheme of interest
    parc_img - Nifti1 image for the parcellation scheme of interest
    '''
    img_dict = {}
    for feat in features: 
        curr_map = np.zeros_like(parc_data)
        for i,roi in enumerate(rois): 
            if scd_effect_dict[mf].loc[roi, 'SCdose_FDR'] < 0.05:
                t_val = scd_effect_dict[mf].loc[roi, 'SCdose_t'] 
                curr_map[parc_data==(i+1)] = t_val
        curr_img = nib.Nifti1Image(curr_map, 
                                   affine=parc_img.affine, 
                                   header=parc_img.header)
        img_dict[feat] = curr_img
    return img_dict

def grad2vols(gradients, atlas, labels, image_dim='4D'):
    """
    Takes computed gradient(s) from fitted GradientMaps object and converts them
    to 3D image(s) in volume format based on the atlas utilized in the GradientMaps
    object. The output can either be one 4D image where 3D gradients are concatenated
    or a list of 3D images, one per gradient.
    ----------
    gradients: array_like
        Fitted GradientMaps object.
    atlas_filename: path or Nifti1Image
        Parcellation utilized within the computation of gradients, that is fmrivols2conn.
        Either name of the corresponding parcellation dataset or loaded dataset.
        Uses nilearn.datasets interface to fetch parcellation datasets and their information.
    labels: list of parcellation labels
    image_dim: str
        Image dimensions of the Nifti1Image output. Either one 4D image where gradient
        maps are concatenated along the 4th dimension ('4D') or a list of 3D images, one per
        gradient ('3D'). Default is '4D'.
    Returns
    -------
    gradient_maps_vol: Nifti1Image
       4D image with gradient maps concatenated along the 4th dimension or list of 3D images, one
       per gradient.
    """

    #check if atlas_filename is str and thus should be loaded
    if isinstance(atlas, str):
        atlas_filename = eval('datasets.fetch_%s()' %atlas).maps
        atlas_img = nib.load(atlas_filename)
        atlas_data = atlas_img.get_fdata()
        labels = eval('datasets.fetch_%s()' %atlas).labels
    else:
    #read and define atlas data, affine and header
        atlas_img = atlas
        atlas_data = atlas.get_fdata()
        labels = labels

    gradient_list = []
    gradient_vols = []

    for gradient in range(gradients.n_components):

        gradient_list.append(np.zeros_like(atlas_data))

    for i, g in enumerate(gradient_list):
        for j in range(0, len(labels)):
            g[atlas_data == (j + 1)] = gradients.gradients_.T[i][j]
        gradient_vols.append(nib.Nifti1Image(g, atlas_img.affine, atlas_img.header))

    if image_dim is None or image_dim=='4D':
        gradient_vols = nib.concat_images(gradient_vols)
        print('Gradient maps will be concatenated within one 4D image.')
    elif image_dim=='3D':
        print('Gradient maps will be provided as a list of 3D images, one per gradient.')

    return gradient_vols

def compute_pc1_overlap_pval(xxy_null_pc1_df, xyy_null_pc1_df, overlap_type="", empirical_overlap=[]):
    num_empirical_null_pc1_overlap = 0
    for i in range(0, 100):
        xxy_rois = []
        xyy_rois = []
        xxy_null_pc1_curr = xxy_null_pc1_df[f'PC1_Null_{i}']
        xyy_null_pc1_curr = xyy_null_pc1_df[f'PC1_Null_{i}']
        r,p = stats.pearsonr(xxy_null_pc1_curr, xyy_null_pc1_curr)
        if r < 0:
            xyy_null_pc1_curr *= -1
        if overlap_type == 'high':
            for roi in hcp_func_rois:
                if xxy_null_pc1_curr[roi] > np.percentile(xxy_null_pc1_curr,95):
                    xxy_rois.append(roi)
                if xyy_null_pc1_curr[roi] > np.percentile(xyy_null_pc1_curr,95):
                    xyy_rois.append(roi)
        elif overlap_type == 'low':
            for roi in hcp_func_rois:
                if xxy_null_pc1_curr[roi] < np.percentile(xxy_null_pc1_curr,5):
                    xxy_rois.append(roi)
                if xyy_null_pc1_curr[roi] < np.percentile(xyy_null_pc1_curr,5):
                    xyy_rois.append(roi)
        elif overlap_type == 'neutral':
            for roi in hcp_func_rois:
                if xxy_null_pc1_curr[roi] > np.percentile(xxy_null_pc1_curr,45) and xxy_null_pc1_curr[roi] < np.percentile(xxy_null_pc1_curr,55):
                    xxy_rois.append(roi)
                if xyy_null_pc1_curr[roi] > np.percentile(xyy_null_pc1_curr,45) and xyy_null_pc1_curr[roi] < np.percentile(xyy_null_pc1_curr,55):
                    xyy_rois.append(roi)
        null_rois_overlap = set(xxy_rois).intersection(xyy_rois)
        len_null_empirical_overlap = len(list(set(null_rois_overlap).intersection(empirical_overlap)))
        if len_null_empirical_overlap > 0:
            num_empirical_null_pc1_overlap += 1    
    pval = num_empirical_null_pc1_overlap*0.01
    return pval



def make_barplot_horizontal(data, netorder, xticklabels, disorders=None,
                            diff_colorbars={},
                            title='', yaxis_label='PC1 (zscore)', fname=None, figsize=None,**kwargs):
    """
    Makes barplot of network z-scores as a function of disorder
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with as least columns ['zscore', 'network', 'parcellation',
        'scale', 'sig']
    netorder : list
        Order in which networks should be plotted within each barplot
    disorder : list, optional
        List of disorders that should be plotted (and the order in which they should
        be plotted)
    fname : str or os.PathLike, optional
        Path to where generated figure should be saved
    kwargs : key-value pairs
        Passed to `ax.set()` on the generated boxplot
    Returns
    -------
    fig : matplotlib.figure.Figure
        Plotted figure
    """

    colors = np.array([np.array([0.7254902, 0.72941176, 0.73333333]),
                       np.array([0.94117647, 0.40784314, 0.41176471])])
    defaults = {'ylabel': '', 'xticklabels': xticklabels, 'xticks': range(0,len(xticklabels)), 
                'ylim': (-3.5, 3.5)}
    defaults.update(kwargs)

    if disorders == None:
        disorders = data['disorder'].unique()
    if figsize == None:
        figsize=((2.125) * len(disorders), 3)
    fig, axes = plt.subplots(1, len(disorders), sharey=True,
                             figsize=figsize)

    # edge case for if we're only plotting one barplot
    if not isinstance(axes, (np.ndarray, list)):
        axes = [axes]

    for n, ax in enumerate(axes):
        # get the data for the relevant disorder
        d = data.query(f'disorder == "{disorders[n]}"')
        if diff_colorbars:
            palette = [diff_colorbars[network] if d[d.network==network]['sig'].values[0]==1 else np.array([0.7254902, 0.72941176, 0.73333333]) for network in netorder]
        else:
            palette = colors[np.asarray([d[d.network==network]['sig'].values[0] for network in netorder])]
        ax = sns.barplot(x='network', y='zscore', data=d,
                         order=netorder, palette=palette, ax=ax)
        lab = '-\n'.join(disorders[n].split('-'))
        ax.set(xlabel=lab, **defaults)
        sns.despine(ax=ax)
        ax.hlines(0, -0.5, len(netorder) - 0.5, linewidth=0.5, ls='--', color='black')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    axes[0].set_ylabel(yaxis_label, labelpad=10)
    
    fig.suptitle(title)
    if fname is not None:
#         fname.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(fname, bbox_inches='tight', transparent=True, dpi=300)
        plt.close(fig=fig)
    return fig


def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)