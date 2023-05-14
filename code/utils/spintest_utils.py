import os
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from scipy import stats, ndimage
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests

from sklearn.preprocessing import MinMaxScaler

# Much of the functionality here has been adapted from https://github.com/netneurolab/markello_spatialnulls

ALPHA=0.05

def _get_permcorr(data, perm):
    """

    Gets max value of correlation between `data` and `perm`
    Excludes diagonal of correlation
    Parameters
    ----------
    data : (R, T) array_like
        Input data where `R` is regions and `T` is neurosynth terms
    perm : (R, T) array_like
        Permuted data where `R` is regions and `T` is neurosynth terms
    Returns
    -------
    corr : float
        Maximum value of correlations between `data` and `perm`
    """

    data, perm = np.asarray(data), np.asarray(perm)

    # don't include NaN data in the correlation process
    mask = np.logical_not(np.all(np.isnan(perm), axis=1))

    # we want to correlat phenotypes across regions, not vice-versa
    data, perm = data[mask].T, perm[mask].T
    out = np.corrcoef(data, perm)

    # grab the upper right quadrant of the resultant correlation matrix and
    # mask the diagonal, then take absolute max correlation
    mask_diag = np.logical_not(np.eye(len(data)), dtype=bool)
    corrs = out[len(data):, :len(data)] * mask_diag

    return np.abs(corrs).max()

def gen_permcorrs(data, spins, fname):
    """
    Generates permuted correlations for `data` with `spins`
    Parameters
    ----------
    data : (R, T) array_like
        Input data where `R` is regions and `T` is neurosynth terms
    spins : (R, P) array_like
        Spin resampling matrix where `R` is regions and `P` is the number of
        resamples
    fname : str or os.PathLike
        Filepath specifying where generated null distribution should be saved
    Returns
    -------
    perms : (P, 1) numpy.ndarray
        Permuted correlations
    """

    data = np.asarray(data)

#     fname = putils.pathify(fname)
#     if fname.exists():
#         return np.loadtxt(fname).reshape(-1, 1)

    if isinstance(spins, (str, os.PathLike)):
        spins = np.loadtxt(spins, delimiter=',', dtype='int32')

    permcorrs = np.zeros((spins.shape[-1], 1))
    for n, spin in enumerate(spins.T):
        msg = f'{n:>5}/{spins.shape[-1]}'
        print(msg, end='\b' * len(msg), flush=True)
        # this will only have False values when spintype == 'baum'
        mask = np.logical_and(spin != -1, np.all(~np.isnan(data), axis=1))
        # get the absolute max correlation from the null correlation matrix
        permcorrs[n] = _get_permcorr(data[mask], data[spin][mask])

#     print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    # save these to disk for later re-use
    #putils.save_dir(fname, permcorrs)

    return permcorrs


def get_fwe(real, perm, alpha=.05):
    """
    Gets p-values from `real` based on null distribution of `perm`
    Parameters
    ----------
    real : (T, T) array_like
        Original correlation matrix (or similar)
    perm : (P, 1) array_like
        Null distribution for `real` based on `P` permutations
    Returns
    -------
    pvals : (T, T) array_like
        Non-parametric p-values for `real`
    """

    real, perm = np.asarray(real), np.asarray(perm)

    if perm.ndim == 1:
        perm = perm.reshape(-1, 1)

    pvals = np.sum(perm >= np.abs(real.flatten()), axis=0) / len(perm)
    pvals = np.reshape(pvals, real.shape)

    thresh = np.sum(np.triu(pvals < alpha, k=1))
    print(f'{thresh:>4} correlation(s) survive FWE-correction')

    return pvals


def _get_netmeans(data, networks, nets=None):
    """
    Gets average of `data` within each label specified in `networks`
    Parameters
    ----------
    data : (N,) array_like
        Data to be averaged within networks
    networks : (N,) array_like
        Network label for each entry in `data`
    Returns
    -------
    means : (L,) numpy.ndarray
        Means of networks
    """

    data, networks = np.asarray(data), np.asarray(networks)
    nparc = np.bincount(networks)[1:]

    if nets is None:
        nets = np.trim_zeros(np.unique(networks))

    mask = np.logical_not(np.isnan(data))

    # otherwise, compute the average T1w/T2w within each network
    data, networks = data[mask], networks[mask]
    with np.errstate(invalid='ignore'):
        permnets = ndimage.mean(data, networks, nets)

    return permnets

def gen_permnets(data, networks, spins):
    """
    Generates permuted network partitions of `data` and `networks` with `spins`
    Parameters
    ----------
    data : (R,) array_like
        Input data where `R` is regions
    networks : (R,) array_like
        Network labels for `R` regions
    spins : (R, P) array_like
        Spin resampling matrix where `R` is regions and `P` is the number of
        resamples
    fname : str or os.PathLike
        Filepath specifying where generated null distribution should be saved
    Returns
    -------
    permnets : (P, L) numpy.ndarray
        Permuted network means for `L` networks
    """

    data, networks = np.asarray(data), np.asarray(networks)

    # if the output file already exists just load that and return it
#     fname = Path(fname)
#     if fname.exists():
#         return np.loadtxt(fname, delimiter=',')

    # if we were given a file for the resampling array, load it
    if isinstance(spins, (str, os.PathLike)):
        spins = simnulls.load_spins(spins, n_perm=10000)

    nets = np.trim_zeros(np.unique(networks))
    permnets = np.full((spins.shape[-1], len(nets)), np.nan)
    for n, spin in enumerate(spins.T):
#         msg = f'{n:>5}/{spins.shape[-1]}'
#         print(msg, end='\b' * len(msg), flush=True)

        spindata = data[spin]
        spindata[spin == -1] = np.nan

        # get the means of each network for each spin
        permnets[n] = _get_netmeans(spindata, networks, nets)

#     print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)
    #putils.save_dir(fname, permnets)

    return permnets

def get_fwe_partitions(real, perm, alpha=ALPHA):
    """
    Gets p-values from `real` based on null distribution of `perm`
    Parameters
    ----------
    real : (1, L) array_like
        Real partition means for `L` networks
    perm : (S, L) array_like
        Null partition means for `S` permutations of `L` networks
    alpha : (0, 1) float, optional
        Alpha at which to check for p-value significance. Default: ALPHA
    Returns
    -------
    zscores : (L,) numpy.ndarray
        Z-scores of `L` networks
    pvals : (L,) numpy.ndarray
        P-values corresponding to `L` networks
    """

    real, perm = np.asarray(real), np.asarray(perm)

    if real.ndim == 1:
        real = real.reshape(1, -1)

    # de-mean distributions to get accurate two-tailed p-values
    permmean = np.nanmean(perm, axis=0, keepdims=True)
    permstd = np.nanstd(perm, axis=0, keepdims=True, ddof=1)
    real -= permmean
    perm -= permmean

    # get z-scores and pvals (add 1 to numerator / denominator for pvals)
    zscores = np.squeeze(real / permstd)
    numerator = np.sum(np.abs(np.nan_to_num(perm)) >= np.abs(real), axis=0)
    denominator = np.sum(np.logical_not(np.isnan(perm)), axis=0)
    pvals = np.squeeze((1 + numerator) / (1 + denominator))

    # print networks with pvals below threshold
    print(', '.join([f'{z:.2f}' if pvals[n] < ALPHA else '0.00'
                     for n, z in enumerate(zscores)]))

    return zscores, pvals

def make_barplot_vertical(data, netorder, xticklabels, disorders=None, fname=None, **kwargs):
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

    defaults = dict(ylabel='', xticklabels=xticklabels, xticks=range(0,len(xticklabels)), ylim=(-3.5, 3.5))
    defaults.update(kwargs)

    if disorders == None:
        disorders = data['disorder'].unique()
    fig, axes = plt.subplots(len(disorders), 1, sharey=True,
                             figsize=(13,18))

    # edge case for if we're only plotting one barplot
    if not isinstance(axes, (np.ndarray, list)):
        axes = [axes]

    for n, ax in enumerate(axes):
        # get the data for the relevant disorder
        d = data.query(f'disorder == "{disorders[n]}"')
        palette = COLORS[np.asarray([d[d.network==network]['sig'].values[0] for network in netorder])]
        # plot!
        ax = sns.barplot(x='network', y='zscore', data=d,
                         order=netorder, palette=palette, ax=ax)
        lab = '-\n'.join(disorders[n].split('-'))
        ax.set(xlabel="", **defaults)
        sns.despine(ax=ax, bottom=True)
        ax.hlines(0, -0.5, len(netorder) - 0.5, linewidth=0.5)
        ax.set_xticklabels([], rotation=90)
        ax.set_title(disorders[n])
        #ax.set_ylabel('PC1 (z)', labelpad=10)
    axes[2].set_xticklabels(xticklabels, rotation=90)
    axes[2].set(xlabel="network", **defaults)
    axes[1].set_ylabel('PC1 (z)', labelpad=10)


    if fname is not None:
#         fname.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close(fig=fig)

    return fig


def make_partitions_specificity_df(data, disorders, spins, 
                                   annotation_name, annotation_codes, 
                                   annotation_order):
    partitions_specifity_df = pd.DataFrame(columns=['zscore', 'network', 'disorder', 'sig'])
    for disorder in disorders:
        real = _get_netmeans(data[disorder], 
                                            data[annotation_name].astype(int))
        permnets = gen_permnets(data[disorder], 
                                               data[annotation_name].astype(int), spins)
        zscores, pvals = get_fwe_partitions(real, permnets)
        for i in range(len(zscores)):
            if pvals[i] < 0.05:
                sig = 1
            else:
                sig = 0
            curr_df = pd.DataFrame([[zscores[i], 
                                     list(annotation_codes.keys())[i], 
                                     disorder.split("_")[0], sig]], 
                                     columns=['zscore', 'network', 'disorder', 'sig'])
            partitions_specifity_df = pd.concat([curr_df, partitions_specifity_df], 
                                                ignore_index=True)
    return partitions_specifity_df

def compute_real_Fstat(data, disorder, annotation):
    mod = smf.ols(formula=f'{disorder} ~ C({annotation})', data=data)
    res = mod.fit()
    return res.fvalue

def compute_null_Fstat(data, spins, disorder, annotation):
    null_fstats = []
    for i in range(0,len(spins.T)):
        new_df = pd.DataFrame(index=data.index, columns=['spin', annotation])
        new_df[annotation] = data[annotation]
        new_df['spin'] = list(data[disorder][spins.T[i]])
        mod = smf.ols(formula=f'spin ~ C({annotation})', data=new_df)
        res = mod.fit()
        null_fstats.append(res.fvalue)
    return null_fstats