# We begin by importing all necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from matplotlib.cm import register_cmap
from random import choices
from collections import Counter
from scipy.linalg import norm as nrm
from sklearn.decomposition import TruncatedSVD as tSVD
import kneed as kn
from PlotFigures import *

# Setting up plotting options and define a custom palette.
rgb = [(1, 1, 1), (73 / 255, 111 / 255, 252 / 255), (252 / 255, 102 / 255, 65 / 255)]
custom2 = sns.blend_palette(rgb, n_colors=255, as_cmap=True)
register_cmap("customBlend", custom2)

font = {'family': 'normal',
        'size': 26}
plt.rc('font', **font)
sns.set_style('white')
plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.rc('text', usetex=False)
np.seterr(divide='ignore', invalid='ignore')

# For each experimental session (there are 39), store all brain regions that were recorded.
b_all = []
b_uni = []
for m in range(39):
    filename = 'brainLocs_session%d.npz' % m
    a = np.load(filename)
    b = np.zeros_like(a)
    for i in range(len(a)):
        b[i] = a[list(a.keys())[i]]
    b_all.append(b)
    b_uni.append(np.unique(b))

# Change any of these to 1 in order to compute shufflings (doShuff) or subsampling (doSubSam) analyses.
# If not, both of them are set to 0 for speed purposes.

doShuff = 0
doSubSam = 0

# How many shufflings (numShuf), repetitions per subsampling (numSubRep) and subsamplings (numSamp) we want to explore. In the
# paper we used 50, 10 and 10, respectively.
numShuf = 50
numSubRep = 10
numSamp = 10

# Preallocation of basic quantities.
measured_dist = []
measured_dist_wrong = []
surrogate_dist = []
surrogate_wrong_dist = []
surrogate_specIdx = []
spec_idx = []
dist_hit = []
dist_miss = []
specIdx_hit = []
specIdx_miss = []

perc_list = []
perc_wrong_list = []

# Preallocation of subsampling quantities.
measured_dist_subsam = []
measured_dist_subsam_wrong = []
spec_idx_subsam = []
dist_hit_subsam = []
dist_miss_subsam = []
specIdx_hit_subsam = []
specIdx_miss_subsam = []

# Loop over experimental sessions (39).
for m in range(39):
    # We load the population vectors we computed in SpikesOverSessions.py
    filename = 'PopVectors_responsive200ms_session%d.npz' % m
    dataa = np.load(filename)

    # Our stimulus feature is the contrast difference between screens.
    filename_stimDiffs = 'StimDifference_session%d.npy' % m
    stimDiffs = np.load(filename_stimDiffs)

    # We remove those trials in which there was a projected image and the contrast difference was null.
    filename_stims = 'Stims_session%d.npz' % m
    stims = np.load(filename_stims)
    stims_left = stims['left']
    stims_right = stims['right']
    stims_left = stims_left.reshape(stims_left.shape[0], )
    stims_right = stims_right.reshape(stims_right.shape[0], )
    idx_filter = ~np.logical_and(stims_left == stims_right, stims_left != 0)

    # We will split quantities by correct and incorrect responses.
    filename_choices = 'Feedbacks_session%d.npy' % m
    choicess = np.load(filename_choices)
    choicess = choicess.reshape(choicess.shape[0], )[idx_filter]
    stimDiffs = stimDiffs.reshape(stimDiffs.shape[0], )[idx_filter]

    # Preallocation of all the previously allocated quantities, but per session in this case. We will erase them after
    # each session and save a list of these over sessions, to pool things together.
    dist_pop = np.nan * np.ones((choicess.shape[0], len(dataa.keys())))
    dist_wrong_pop = np.nan * np.ones((choicess.shape[0], len(dataa.keys())))

    dist_Sh = np.nan * np.ones((choicess.shape[0], len(dataa.keys()), numShuf))
    dist_wrong_Sh = np.nan * np.ones((choicess.shape[0], len(dataa.keys()), numShuf))
    specIdx_Sh = np.nan * np.ones((choicess.shape[0], len(dataa.keys()), numShuf))
    perc = np.nan * np.ones((choicess.shape[0], len(dataa.keys())))
    perc_wrong = np.nan * np.ones((choicess.shape[0], len(dataa.keys())))

    dist_subsam = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys()), numSubRep, numSamp))
    dist_wrong_subsam = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys()), numSubRep, numSamp))

    # For this particular session, loop over all recorded areas, excluding "root" (as it's an undefined region
    # in the Neuropixel probe).
    for bb in range(len(np.unique(b_all[m]))):
        if np.unique(b_all[m])[bb] != 'root':
            num_cells = dataa[list(dataa.keys())[bb]].shape[1]
            popVectors = dataa[list(dataa.keys())[bb]][idx_filter, :]

            # Check if there is more than one recorded neuron in that area.
            if popVectors.shape[1] > 1:
                # In order to automatize the PCA calculations over areas with a different number of recorded neurons,
                # we find the knee (elbow) in the variance explained over number of components.
                nComps = min(popVectors.shape[1] - 1, popVectors.shape[0] - 1)
                svdd = tSVD(n_components=nComps)
                a = sts.zscore(popVectors)
                a[np.isnan(a)] = 0
                x2 = svdd.fit_transform(a)
                knee = kn.KneeLocator(np.arange(0, nComps), svdd.explained_variance_ratio_,
                                      curve='convex', direction='decreasing', interp_method='interp1d')
                # Safety checks.
                if knee.knee is not None:
                    if knee.knee > 0:
                        nComps2 = knee.knee[knee.knee > 0][0]
                    else:
                        nComps2 = 1
                else:
                    nComps2 = 1
                svdd2 = tSVD(n_components=nComps2)
                a = sts.zscore(popVectors)
                a[np.isnan(a)] = 0
                x = svdd2.fit_transform(a)
                # Templates.
                a_left = np.dot(svdd2.components_, np.nanmean(popVectors[stimDiffs < 0, :], axis=0))
                a_right = np.dot(svdd2.components_, np.nanmean(popVectors[stimDiffs > 0, :], axis=0))
                # As the Euclidean distance is not upper-bounded, we normalize all quantities by the distance between
                # templates in this PCA space.
                d_ref = nrm(a_left - a_right)

                # Loop over trials
                for i in range(popVectors.shape[0]):
                    if stimDiffs[i] < 0:
                        dist_pop[i, bb] = nrm(np.dot(svdd2.components_, popVectors[i, :]) - a_left) / d_ref
                        dist_wrong_pop[i, bb] = nrm(np.dot(svdd2.components_, popVectors[i, :]) - a_right) / d_ref
                        if doSubSam:
                            # Loop over subsampling sizes
                            for k in range(numSamp):
                                # We won't perform subsampling analyses with less than 20 neurons.
                                if popVectors.shape[1] > 20:
                                    # Loop over number of subsampling realizations.
                                    for s in range(numSubRep):
                                        subsam = np.random.choice(popVectors.shape[1],
                                                                  int(np.floor(popVectors.shape[1] / (k + 1))))
                                        nComps = min(subsam.shape[0] - 1, popVectors.shape[0] - 1)
                                        if nComps > 1:
                                            svdd3 = tSVD(n_components=nComps)
                                            a = sts.zscore(popVectors[:, subsam])
                                            a[np.isnan(a)] = 0
                                            x2 = svdd3.fit_transform(a)
                                            knee = kn.KneeLocator(np.arange(0, nComps), svdd3.explained_variance_ratio_,
                                                                  curve='convex', direction='decreasing',
                                                                  interp_method='interp1d')
                                            if knee.knee is not None:
                                                if knee.knee > 0:
                                                    nComps2 = knee.knee[knee.knee > 0][0]
                                                else:
                                                    nComps2 = 1
                                            else:
                                                nComps2 = 1
                                            svdd4 = tSVD(n_components=nComps2)
                                            a = sts.zscore(popVectors[:, subsam])
                                            a[np.isnan(a)] = 0
                                            x = svdd4.fit_transform(a)
                                            b_left = popVectors[stimDiffs < 0, :]
                                            b_right = popVectors[stimDiffs > 0, :]
                                            a_leftt = np.dot(svdd4.components_, np.nanmean(b_left[:, subsam], axis=0))
                                            a_rightt = np.dot(svdd4.components_, np.nanmean(b_right[:, subsam], axis=0))
                                            d_ref = nrm(a_leftt - a_rightt)
                                            singletrial = popVectors[i, :]

                                            dist_subsam[i, bb, s, k] = nrm(
                                                np.dot(svdd4.components_, singletrial[subsam]) - a_leftt) / d_ref
                                            dist_wrong_subsam[i, bb, s, k] = nrm(
                                                np.dot(svdd4.components_, singletrial[subsam]) - a_rightt) / d_ref

                        if doShuff:
                            # Shuffling loop.
                            for s in range(numShuf):
                                aa = choices(np.arange(0, a_left.shape[0], 1), a_left / np.sum(a_left),
                                             k=int(np.sum(popVectors[i, :])))
                                rands = Counter(aa)
                                ww = np.zeros_like(a_left)
                                ww[list(rands.keys())] = list(rands.values())

                                dist_Sh[i, bb, s] = nrm(ww - a_left) / d_ref
                                dist_wrong_Sh[i, bb, s] = nrm(ww - a_right) / d_ref
                                specIdx_Sh[i, bb, s] = dist_wrong_Sh[i, bb, s] - dist_Sh[i, bb, s]

                            perc[i, bb] = sts.percentileofscore(dist_Sh[i, bb, :], dist_pop[i, bb]) / 100
                            perc_wrong[i, bb] = sts.percentileofscore(dist_wrong_Sh[i, bb, :],
                                                                      dist_wrong_pop[i, bb]) / 100

                    elif stimDiffs[i] > 0:
                        dist_pop[i, bb] = nrm(np.dot(svdd2.components_, popVectors[i, :]) - a_right) / d_ref
                        dist_wrong_pop[i, bb] = nrm(np.dot(svdd2.components_, popVectors[i, :]) - a_left) / d_ref
                        if doSubSam:
                            # Loop over subsampling sizes
                            for k in range(numSamp):
                                # We won't perform subsampling analyses with less than 20 neurons.
                                if popVectors.shape[1] > 20:
                                    # Loop over number of subsampling realizations.
                                    for s in range(numSubRep):
                                        subsam = np.random.choice(popVectors.shape[1],
                                                                  int(np.floor(popVectors.shape[1] / (k + 1))))
                                        if np.sum(popVectors[:, subsam]) != 0:
                                            nComps = min(subsam.shape[0] - 1, popVectors.shape[0] - 1)
                                            if nComps > 1:
                                                svdd5 = tSVD(n_components=nComps)
                                                a = sts.zscore(popVectors[:, subsam])
                                                a[np.isnan(a)] = 0
                                                x2 = svdd5.fit_transform(a)
                                                knee = kn.KneeLocator(np.arange(0, nComps),
                                                                      svdd5.explained_variance_ratio_,
                                                                      curve='convex', direction='decreasing',
                                                                      interp_method='interp1d')
                                                if knee.knee is not None:
                                                    if knee.knee > 0:
                                                        nComps2 = knee.knee[knee.knee > 0][0]
                                                    else:
                                                        nComps2 = 1
                                                else:
                                                    nComps2 = 1
                                                svdd6 = tSVD(n_components=nComps2)
                                                a = sts.zscore(popVectors[:, subsam])
                                                a[np.isnan(a)] = 0
                                                x = svdd6.fit_transform(a)
                                                b_left = popVectors[stimDiffs < 0, :]
                                                b_right = popVectors[stimDiffs > 0, :]
                                                if np.logical_and(np.sum(np.nanmean(b_left[:, subsam], axis=0)) > 0,
                                                                  np.sum(np.nanmean(b_right[:, subsam], axis=0)) > 0):
                                                    a_leftt = np.dot(svdd6.components_,
                                                                     np.nanmean(b_left[:, subsam], axis=0))
                                                    a_rightt = np.dot(svdd6.components_,
                                                                      np.nanmean(b_right[:, subsam], axis=0))
                                                    d_ref = nrm(a_leftt - a_rightt)

                                                    singletrial = popVectors[i, :]

                                                    dist_subsam[i, bb, s, k] = nrm(np.dot(svdd6.components_,
                                                                                          singletrial[
                                                                                              subsam]) - a_rightt) / d_ref
                                                    dist_wrong_subsam[i, bb, s, k] = nrm(np.dot(svdd6.components_,
                                                                                                singletrial[
                                                                                                    subsam]) - a_leftt) / d_ref
                                                else:
                                                    dist_subsam[i, bb, s, k] = np.nan
                                                    dist_wrong_subsam[i, bb, s, k] = np.nan

                        if doShuff:
                            # Shuffling loop.
                            for s in range(numShuf):
                                aa = choices(np.arange(0, a_right.shape[0], 1), a_right / np.sum(a_right),
                                             k=int(np.sum(popVectors[i, :])))
                                rands = Counter(aa)
                                ww = np.zeros_like(a_right)
                                ww[list(rands.keys())] = list(rands.values())

                                dist_Sh[i, bb, s] = nrm(ww - a_right) / d_ref
                                dist_wrong_Sh[i, bb, s] = nrm(ww - a_left) / d_ref
                                specIdx_Sh[i, bb, s] = dist_wrong_Sh[i, bb, s] - dist_Sh[i, bb, s]

                            perc[i, bb] = sts.percentileofscore(dist_Sh[i, bb, :], dist_pop[i, bb]) / 100
                            perc_wrong[i, bb] = sts.percentileofscore(dist_wrong_Sh[i, bb, :],
                                                                      dist_wrong_pop[i, bb]) / 100

                    elif stimDiffs[i] == 0:
                        dist_pop[i, bb] = np.nan

    measured_dist.append(dist_pop)
    measured_dist_wrong.append(dist_wrong_pop)
    specIdx_hit.append(dist_wrong_pop[choicess == 1, :] - dist_pop[choicess == 1, :])
    specIdx_miss.append(dist_wrong_pop[choicess == -1, :] - dist_pop[choicess == -1, :])
    dist_hit.append(dist_pop[choicess == 1, :])
    dist_miss.append(dist_pop[choicess == -1, :])
    spec_idx.append(dist_wrong_pop - dist_pop)
    if doShuff:
        surrogate_dist.append(dist_Sh)
        surrogate_wrong_dist.append(dist_wrong_Sh)
        surrogate_specIdx.append(specIdx_Sh)
        perc_list.append(perc)
        perc_wrong_list.append(perc_wrong)
    if doSubSam:
        measured_dist_subsam.append(dist_subsam)
        measured_dist_subsam_wrong.append(dist_wrong_subsam)
        spec_idx_subsam.append(dist_wrong_subsam - dist_subsam)
        specIdx_hit_subsam.append(dist_wrong_subsam[choicess == 1, :, :, :] - dist_subsam[choicess == 1, :, :, :])
        specIdx_miss_subsam.append(dist_wrong_subsam[choicess == -1, :, :, :] - dist_subsam[choicess == -1, :, :, :])
        dist_hit_subsam.append(dist_subsam[choicess == 1, :, :, :])
        dist_miss_subsam.append(dist_subsam[choicess == -1, :, :, :])
    print('m = ', m)

# Reorganize all quantities: pool together over sessions, for all areas.
flat_b = [item for sublist in b_uni for item in sublist]
uni_flat = np.unique(flat_b)

dist_fin = []
dist_wrong_fin = []
spec_idx_fin = []
dist_hit_fin = []
dist_miss_fin = []
specIdx_hit_fin = []
specIdx_miss_fin = []

dist_Sh_fin = []
specIdx_Sh_fin = []
dist_wrong_Sh_fin = []
perc_fin = []
perc_wrong_fin = []

for bb in range(len(uni_flat)):
    meas_tmp = []
    measured_wrong_tmp = []
    perc_tmp = []
    perc_wrong_tmp = []
    surr_tmp = []
    specIdx_surr_tmp = []
    surr_wrong_tmp = []
    spec_idx_tmp = []
    dist_hit_tmp = []
    dist_miss_tmp = []
    specIdx_hit_tmp = []
    specIdx_miss_tmp = []

    for m in range(39):
        if np.logical_and(uni_flat[bb] in b_uni[m], uni_flat[bb] != 'root'):
            idx = np.where(b_uni[m] == uni_flat[bb])[0]
            if doShuff:
                surr_tmp.append(surrogate_dist[m][:, idx[0], :].ravel())
                surr_wrong_tmp.append(surrogate_wrong_dist[m][:, idx[0]].ravel())
                specIdx_surr_tmp.append(surrogate_specIdx[m][:, idx[0]].ravel())
                perc_tmp.append(perc_list[m][:, idx[0]])
                perc_wrong_tmp.append(perc_wrong_list[m][:, idx[0]])

            meas_tmp.append(measured_dist[m][:, idx[0]])
            measured_wrong_tmp.append(measured_dist_wrong[m][:, idx[0]])
            dist_hit_tmp.append(dist_hit[m][:, idx[0]])
            dist_miss_tmp.append(dist_miss[m][:, idx[0]])
            spec_idx_tmp.append(spec_idx[m][:, idx[0]])
            specIdx_hit_tmp.append(specIdx_hit[m][:, idx[0]])
            specIdx_miss_tmp.append(specIdx_miss[m][:, idx[0]])

    dist_fin.append([item for sublist in meas_tmp for item in sublist])
    dist_wrong_fin.append([item for sublist in measured_wrong_tmp for item in sublist])
    dist_Sh_fin.append([item for sublist in surr_tmp for item in sublist])
    dist_wrong_Sh_fin.append([item for sublist in surr_wrong_tmp for item in sublist])
    specIdx_Sh_fin.append([item for sublist in specIdx_surr_tmp for item in sublist])
    perc_fin.append([item for sublist in perc_tmp for item in sublist])
    perc_wrong_fin.append([item for sublist in perc_wrong_tmp for item in sublist])
    spec_idx_fin.append([item for sublist in spec_idx_tmp for item in sublist])
    specIdx_hit_fin.append([item for sublist in specIdx_hit_tmp for item in sublist])
    specIdx_miss_fin.append([item for sublist in specIdx_miss_tmp for item in sublist])
    dist_hit_fin.append([item for sublist in dist_hit_tmp for item in sublist])
    dist_miss_fin.append([item for sublist in dist_miss_tmp for item in sublist])

if doSubSam:
    # Same, but for the subsampling loops:
    measured_subsam_fin = []
    measured_subsam_wrong_fin = []
    spec_idx_subsam_fin = []
    dist_hit_subsam_fin = []
    dist_miss_subsam_fin = []
    specIdx_hit_subsam_fin = []
    specIdx_miss_subsam_fin = []
    for bb in range(len(uni_flat)):
        subsam_tmp = []
        subsam_wrong_tmp = []
        spec_idx_subsam_tmp = []
        dist_hit_subsam_tmp = []
        dist_miss_subsam_tmp = []
        specIdx_hit_subsam_tmp = []
        specIdx_miss_subsam_tmp = []
        for k in range(numSamp):
            subsam_tmp_tmp = []
            subsam_wrong_tmp_tmp = []
            spec_idx_subsam_tmp_tmp = []
            dist_hit_subsam_tmp_tmp = []
            dist_miss_subsam_tmp_tmp = []
            specIdx_hit_subsam_tmp_tmp = []
            specIdx_miss_subsam_tmp_tmp = []
            for m in range(39):
                if np.logical_and(uni_flat[bb] in b_uni[m], uni_flat[bb] != 'root'):
                    idx = np.where(b_uni[m] == uni_flat[bb])[0]
                    subsam_tmp_tmp.append(measured_dist_subsam[m][:, idx[0], :, k].ravel())
                    subsam_wrong_tmp_tmp.append(measured_dist_subsam_wrong[m][:, idx[0], :, k].ravel())
                    dist_hit_subsam_tmp_tmp.append(dist_hit_subsam[m][:, idx[0], :, k].ravel())
                    dist_miss_subsam_tmp_tmp.append(dist_miss_subsam[m][:, idx[0], :, k].ravel())
                    spec_idx_subsam_tmp_tmp.append(spec_idx_subsam[m][:, idx[0], :, k].ravel())
                    specIdx_hit_subsam_tmp_tmp.append(specIdx_hit_subsam[m][:, idx[0], :, k].ravel())
                    specIdx_miss_subsam_tmp_tmp.append(specIdx_miss_subsam[m][:, idx[0], :, k].ravel())
            subsam_tmp.append(subsam_tmp_tmp)
            subsam_wrong_tmp.append(subsam_wrong_tmp_tmp)
            dist_hit_subsam_tmp.append(dist_hit_subsam_tmp_tmp)
            dist_miss_subsam_tmp.append(dist_miss_subsam_tmp_tmp)
            spec_idx_subsam_tmp.append(spec_idx_subsam_tmp_tmp)
            specIdx_hit_subsam_tmp.append(specIdx_hit_subsam_tmp_tmp)
            specIdx_miss_subsam_tmp.append(specIdx_miss_subsam_tmp_tmp)
        # Area, subsample step, no. of sessions in which it was recorded, no. of repetitions x no. trials
        measured_subsam_fin.append(subsam_tmp)
        measured_subsam_wrong_fin.append(subsam_wrong_tmp)
        spec_idx_subsam_fin.append(spec_idx_subsam_tmp)
        specIdx_hit_subsam_fin.append(specIdx_hit_subsam_tmp)
        specIdx_miss_subsam_fin.append(specIdx_miss_subsam_tmp)
        dist_hit_subsam_fin.append(dist_hit_subsam_tmp)
        dist_miss_subsam_fin.append(dist_miss_subsam_tmp)

# Convert all lists into DataFrames, with the appropriate column-labelling (71 areas at the end).

df_measured = pd.DataFrame(dist_fin[:-1]).T
df_measured.columns = uni_flat[:-1]
df_measured.drop('OT', axis=1, inplace=True)
df_measured_wrong = pd.DataFrame(dist_wrong_fin[:-1]).T
df_measured_wrong.columns = uni_flat[:-1]
df_measured_wrong.drop('OT', axis=1, inplace=True)
df_dist_hit = pd.DataFrame(dist_hit_fin[:-1]).T
df_dist_hit.columns = uni_flat[:-1]
df_dist_hit.drop('OT', axis=1, inplace=True)
df_dist_miss = pd.DataFrame(dist_miss_fin[:-1]).T
df_dist_miss.columns = uni_flat[:-1]
df_dist_miss.drop('OT', axis=1, inplace=True)
df_spec_idx = pd.DataFrame(spec_idx_fin[:-1]).T
df_spec_idx.columns = uni_flat[:-1]
df_spec_idx.drop('OT', axis=1, inplace=True)
df_specIdx = pd.DataFrame(spec_idx_fin[:-1]).T
df_specIdx.columns = uni_flat[:-1]
df_specIdx.drop('OT', axis=1, inplace=True)
df_specIdx_hit = pd.DataFrame(specIdx_hit_fin[:-1]).T
df_specIdx_hit.columns = uni_flat[:-1]
df_specIdx_hit.drop('OT', axis=1, inplace=True)
df_specIdx_miss = pd.DataFrame(specIdx_miss_fin[:-1]).T
df_specIdx_miss.columns = uni_flat[:-1]
df_specIdx_miss.drop('OT', axis=1, inplace=True)

if doShuff:
    df_surr = pd.DataFrame(dist_Sh_fin[:-1]).T
    df_surr.columns = uni_flat[:-1]
    df_surr.drop('OT', axis=1, inplace=True)
    df_surr_wrong = pd.DataFrame(dist_wrong_Sh_fin[:-1]).T
    df_surr_wrong.columns = uni_flat[:-1]
    df_surr_wrong.drop('OT', axis=1, inplace=True)
    df_surr_specIdx = pd.DataFrame(specIdx_Sh_fin[:-1]).T
    df_surr_specIdx.columns = uni_flat[:-1]
    df_surr_specIdx.drop('OT', axis=1, inplace=True)
    df_perc = pd.DataFrame(perc_fin[:-1]).T
    df_perc.columns = uni_flat[:-1]
    df_perc.drop('OT', axis=1, inplace=True)
    df_perc_wrong = pd.DataFrame(perc_wrong_fin[:-1]).T
    df_perc_wrong.columns = uni_flat[:-1]
    df_perc_wrong.drop('OT', axis=1, inplace=True)

# Load the choice and stimulus information quantities for each area over sessions (from the decoder analyses).
df_choice = pd.read_csv('df_test_choiceInfo_allResponsive_200ms.csv')
df_choice.drop('Unnamed: 0', axis=1, inplace=True)
df_stim = pd.read_csv('df_test_stimInfo_allResponsive_200ms.csv')
df_stim.drop('Unnamed: 0', axis=1, inplace=True)

# From the experimental datapoints, we construct the corresponding continuous distribution via Gaussian Kernel Estimation.
density_stim = sts.kde.gaussian_kde(df_stim.median())
density_stim = density_stim(np.unique(df_stim.median()))

density_choi = sts.kde.gaussian_kde(df_choice.median())
density_choi = density_choi(np.unique(df_choice.median()))

# Using the elbow method to select the stimulus-, choice- or both-informative areas.
knee_stim = kn.KneeLocator(np.unique(df_stim.median())[np.where(density_stim == np.max(density_stim))[0][0] + 10:],
                           density_stim[np.where(density_stim == np.max(density_stim))[0][0] + 10:],
                           curve='convex', direction='decreasing', interp_method='interp1d')

knee_choi = kn.KneeLocator(np.unique(df_choice.median())[np.where(density_choi == np.max(density_choi))[0][0] + 10:],
                           density_choi[np.where(density_choi == np.max(density_choi))[0][0] + 10:],
                           curve='convex', direction='decreasing', interp_method='interp1d')

inf_stim = knee_stim.knee
inf_choi = knee_choi.knee

# As mentioned in Fig. 2, for all areas above the corresponding thresholds (knees of the distribution), we will consider
# them to be stimulus-informative, choice-informative or both-informative.
df_rel = pd.DataFrame(np.ones(df_choice.shape[1]).T)
df_rel = df_rel.T
df_rel.columns = df_choice.columns
for col_id in df_choice.columns:
    if np.logical_and(df_stim[col_id].median() >= inf_stim, df_choice[col_id].median() < inf_choi):
        df_rel[col_id] = 'a_Visual'
    elif np.logical_and(df_stim[col_id].median() < inf_stim, df_choice[col_id].median() >= inf_choi):
        df_rel[col_id] = 'c_Choice'
    elif np.logical_and(df_stim[col_id].median() >= inf_stim, df_choice[col_id].median() >= inf_choi):
        df_rel[col_id] = 'b_Both'
    elif col_id in ['ACB', 'SI', 'EPd']:
        df_rel[col_id] = 'u_Unrelated'

# We will now just work with those areas that are task-related and the 3 other examples (ACB, SI and EPd).
selected = df_rel.values != 1
df_rel_fin = df_rel.iloc[:, selected[0]]
aa = np.argsort(df_rel_fin)
b = df_rel_fin.iloc[:, aa.values[0]]
b_vis = b.reindex(sorted(b.columns[b.values[0] == 'a_Visual']), axis=1)
b_both = b.reindex(sorted(b.columns[b.values[0] == 'b_Both']), axis=1)
b_choi = b.reindex(sorted(b.columns[b.values[0] == 'c_Choice']), axis=1)
b_unrel = b.reindex(sorted(b.columns[b.values[0] == 'u_Unrelated']), axis=1)
b_fin = pd.concat([b_vis, b_both, b_choi, b_unrel], axis=1)

# Out of all of the recorded regions and subregions, we just extract the selected ones, for all quantities.
# We then relabel the columns of all DataFrames, for better inspection and plotting purposes.
df_measured_fin = df_measured.iloc[:, selected[0]]
df_wrong_fin = df_measured_wrong.iloc[:, selected[0]]
df_specIdx_fin = df_spec_idx.iloc[:, selected[0]]
df_hit_fin = df_dist_hit.iloc[:, selected[0]]
df_miss_fin = df_dist_miss.iloc[:, selected[0]]
df_spec_idx_fin = df_spec_idx.iloc[:, selected[0]]
df_specIdx_hit_fin = df_specIdx_hit.iloc[:, selected[0]]
df_specIdx_miss_fin = df_specIdx_miss.iloc[:, selected[0]]

df_measured_fin2 = df_measured_fin[b_fin.columns]
df_wrong_fin2 = df_wrong_fin[b_fin.columns]
df_hit_fin2 = df_hit_fin[b_fin.columns]
df_miss_fin2 = df_miss_fin[b_fin.columns]
df_spec_idx_fin2 = df_spec_idx_fin[b_fin.columns]
df_specIdx_hit_fin2 = df_specIdx_hit_fin[b_fin.columns]
df_specIdx_miss_fin2 = df_specIdx_miss_fin[b_fin.columns]

if doShuff:
    df_perc_fin = df_perc.iloc[:, selected[0]]
    df_perc_wrong_fin = df_perc_wrong.iloc[:, selected[0]]
    df_surr_fin = df_surr.iloc[:, selected[0]]
    df_surr_wrong_fin = df_surr_wrong.iloc[:, selected[0]]
    df_surr_specIdx_fin = df_surr_specIdx.iloc[:, selected[0]]

    df_perc_fin2 = df_perc_fin[b_fin.columns]
    df_perc_wrong_fin2 = df_perc_wrong_fin[b_fin.columns]
    df_surr_fin2 = df_surr_fin[b_fin.columns]
    df_surr_wrong_fin2 = df_surr_wrong_fin[b_fin.columns]
    df_surr_specIdx_fin2 = df_surr_specIdx_fin[b_fin.columns]

# U-tests between different quantities (change u and v).
sign_2 = []
pval_2 = []
v1 = []
v2 = []
for i in range(df_measured_fin2.shape[1]):
    u = df_measured_fin2.iloc[:, i]
    v = df_wrong_fin2.iloc[:, i]
    s, p = sts.mannwhitneyu(u, v, alternative='greater')
    sign_2.append(s)
    pval_2.append(p)
print(sign_2)
print(pval_2)

# If there are multiple comparisons to be corrected for, use a FDR Benjamini-Hochberg procedure, with a FWER of 0.05:
doMulti = 0
if doMulti:
    from statsmodels.stats.multitest import multipletests as mt

    mt(np.array(pval_2), alpha=0.05, method='fdr_bh')

## -------------------- Plotting part. --------------------

# Color code that we use throughout the paper: stim-informative areas in shades of blue, choice-informative in red,
# both-informative in purple and uninformative in gray.
colors = [sns.desaturate('blue', 0.2), sns.desaturate('red', 0.3), sns.desaturate('purple', 0.2),
          sns.desaturate('gray', 0.3)]
colors2 = [sns.desaturate('blue', 0.9), sns.desaturate('red', 0.7), sns.desaturate('purple', 0.9),
           sns.desaturate('black', 0.5)]

if doShuff:
    fig, ax = plt.subplots(nrows=1)
    plotBoxes(ax, df_spec_idx_fin2, df_surr_specIdx_fin2, b_fin, colors, colors2, 1)
    # plt.ylim([0, 25])

data1 = df_measured_fin2
data2 = df_wrong_fin2
fig, ax = plt.subplots(figsize=(12, 8))
plotViolin(ax, data1, data2, b_fin, colors, colors2, 'CorrWrong', plotSpec = 0)

data1 = df_hit_fin2
data2 = df_miss_fin2
fig, ax = plt.subplots(figsize=(12, 8))
plotViolin(ax, data1, data2, b_fin, colors, colors2, 'HitMiss', plotSpec = 0)

fig, ax = plt.subplots(figsize=(8, 1))
plotEffSizes(ax, data1, data2, b_fin, colors, colors2)

greater1 = []
greater2 = []
for i in range(data1.shape[1]):
    a = np.sum(data1.iloc[:, i] >= 0) / np.sum(~np.isnan(data1.iloc[:, i])) * 100
    a2 = np.sum(data2.iloc[:, i] >= 0) / np.sum(~np.isnan(data2.iloc[:, i])) * 100
    greater1.append(a)
    greater2.append(a2)

plt.figure(figsize=(8, 1))
for i in range(data1.shape[1]):
    if b_fin.iloc[:, i].values == 'a_Visual':
        color = colors[0]
        color2 = colors2[0]
    elif b_fin.iloc[:, i].values == 'c_Choice':
        color = colors[1]
        color2 = colors2[1]
    elif b_fin.iloc[:, i].values[0] == 'b_Both':
        color = colors[2]
        color2 = colors2[2]
    elif b_fin.iloc[:, i].values[0] == 'u_Unrelated':
        color = colors[3]
        color2 = colors2[3]
    plt.plot(i, greater1[i], marker='o', alpha=0.5, color=color2, markerfacecolor=color, markersize=6)
    plt.plot(i + 0.3, greater2[i], marker='o', alpha=0.5, color=color, markerfacecolor=color2, markersize=6)
    plt.plot([i, i + 0.3], [greater1[i], greater2[i]], color=color)

plt.xticks(ticks=np.arange(0.2, data1.shape[1] + 0.125, 1), labels=data1.columns, rotation=30, fontsize=16)
plt.axhline(50, lw=2, ls='--', color='gray', alpha=0.5)
plt.ylabel('% of positive points')
plt.ylim([0, 100])

# Subsampling loops and plotting part:
if doSubSam:
    measured_subsam_fin2 = []
    measured_subsam_wrong_fin2 = []
    spec_idx_subsam_fin2 = []
    dist_hit_subsam_fin2 = []
    dist_miss_subsam_fin2 = []
    specIdx_hit_subsam_fin2 = []
    specIdx_miss_subsam_fin2 = []
    for bb in range(len(uni_flat)):
        if uni_flat[bb] in b_fin.columns:
            subsam_tmp = []
            subsam_wrong_tmp = []
            spec_idx_subsam_tmp = []
            dist_hit_subsam_tmp = []
            dist_miss_subsam_tmp = []
            specIdx_hit_subsam_tmp = []
            specIdx_miss_subsam_tmp = []
            for k in range(numSamp):
                subsam_tmp.append([item for sublist in measured_subsam_fin[bb][k] for item in sublist])
                subsam_wrong_tmp.append([item for sublist in measured_subsam_wrong_fin[bb][k] for item in sublist])
                dist_hit_subsam_tmp.append([item for sublist in dist_hit_subsam_fin[bb][k] for item in sublist])
                dist_miss_subsam_tmp.append([item for sublist in dist_miss_subsam_fin[bb][k] for item in sublist])
                spec_idx_subsam_tmp.append([item for sublist in spec_idx_subsam_fin[bb][k] for item in sublist])
                specIdx_hit_subsam_tmp.append([item for sublist in specIdx_hit_subsam_fin[bb][k] for item in sublist])
                specIdx_miss_subsam_tmp.append([item for sublist in specIdx_miss_subsam_fin[bb][k] for item in sublist])

            measured_subsam_fin2.append(subsam_tmp)
            measured_subsam_wrong_fin2.append(subsam_wrong_tmp)
            spec_idx_subsam_fin2.append(spec_idx_subsam_tmp)
            specIdx_hit_subsam_fin2.append(specIdx_hit_subsam_tmp)
            specIdx_miss_subsam_fin2.append(specIdx_miss_subsam_tmp)
            dist_hit_subsam_fin2.append(dist_hit_subsam_tmp)
            dist_miss_subsam_fin2.append(dist_miss_subsam_tmp)

    # If you computed the subsampling quantities, this part plots the layout that can be found in the paper (Fig. 6).
    v_list = [measured_subsam_fin2, measured_subsam_wrong_fin2, dist_hit_subsam_fin2, dist_miss_subsam_fin2,
              specIdx_hit_subsam_fin2, specIdx_miss_subsam_fin2]

    fig, axes = plt.subplots(nrows=4, ncols=12)
    plotSubsam(axes, v_list, spec_idx_subsam_fin2, b_fin, colors, colors2, numSamp)
