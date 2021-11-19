# We begin by importing all necessary libraries.
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.cm import register_cmap
from random import choices
from collections import Counter
from matplotlib import rc
import kneed as kn
from PlotFigures import *

# Setting up plotting options and define a custom palette.
font = {'family' : 'normal',
        'size'   : 22}
rc('font', **font)

rgb = [(1,1,1), ( 73/255, 111/255, 252/255 ), ( 252/255, 102/255, 65/255 )]
custom2 = sns.blend_palette(rgb, n_colors=255, as_cmap=True)
register_cmap("customBlend", custom2)
sns.set_style('white')
plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.rc('text', usetex=False)
np.seterr(divide='ignore', invalid='ignore')

# For each experimental session (there are 39), store all brain regions that were recorded.
b_all = []
b_uni = []
for m in range(39):
    filename = 'brainLocs_session%d.npz'%m
    a = np.load(filename)
    b = np.zeros_like(a)
    for i in range(len(a)):
        b[i] = a[list(a.keys())[i]]
    b_all.append(b)
    b_uni.append(np.unique(b))

# Change any of these to 1 in order to compute shufflings (doShuff), subsampling (doSubSam), one-neuron removal (doRemov),
# or Fano Factor (doFano) analyses. If not, all of them are set to 0 for speed purposes.

doShuff = 1
doSubSam = 0
doRemov = 0
doFano = 0

# How many shufflings (numShuf), repetitions per subsampling (numSubRep) and subsamplings (numSamp) we want to explore. In the
# paper we used 50, 50 and 10, respectively.
numShuf = 50
numSubRep = 50
numSamp = 10

# Preallocation of basic quantities.
measured_corr = []
measured_corr_wrong = []
surrogate_corr = []
surrogate_wrong_corr = []
surrogate_specIdx = []
perc_list = []
perc_wrong_list = []
spec_idx = []
corr_hit = []
corr_miss = []
specIdx_hit = []
specIdx_miss = []
fanoo = []

# Preallocation of subsampling quantities.
measured_corr_subsam = []
measured_corr_subsam_wrong = []
spec_idx_subsam = []
corr_hit_subsam = []
corr_miss_subsam = []
specIdx_hit_subsam = []
specIdx_miss_subsam = []

# Preallocation of one-neuron removal quantities.
cor_remov = []
cor_remov_hit = []
cor_remov_miss = []
specIdx_remov = []
specIdx_remov_hit = []
specIdx_remov_miss = []

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

    cor_pop = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys())))
    cor_wrong_pop = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys())))
    cor_Sh = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys()),numShuf))
    cor_wrong_Sh = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys()),numShuf))
    specIdx_Sh = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys()),numShuf))
    cor_subsam = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys()), numSubRep, numSamp))
    cor_wrong_subsam = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys()), numSubRep, numSamp))

    measured_tmp = []
    measured_wrong_tmp = []
    surrogate_tmp = []
    surrogate_wrong_tmp = []
    spec_idx_tmp = []
    corr_hit_tmp = []
    corr_miss_tmp = []
    specIdx_hit_tmp = []
    specIdx_miss_tmp = []
    eff_tmp_pop = []
    fano_tmp = []
    perc_tmp = []
    perc_wrong_tmp = []
    perc = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys())))
    perc_wrong = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys())))

    cor_remov_tmp = []
    cor_remov_hit_tmp = []
    cor_remov_miss_tmp = []
    specIdx_remov_tmp = []
    specIdx_remov_hit_tmp = []
    specIdx_remov_miss_tmp = []

    # For this particular session, loop over all recorded areas, excluding "root" (as it's an undefined region
    # in the Neuropixel probe).
    for bb in range(len(np.unique(b_all[m]))):
        if np.unique(b_all[m])[bb] != 'root':
            num_cells = dataa[list(dataa.keys())[bb]].shape[1]
            popVectors = dataa[list(dataa.keys())[bb]][idx_filter,:]
            eff_size_perNeu = np.zeros(num_cells)
            cor = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys())))
            cor_wrong = np.nan * np.ones((np.sum(idx_filter), len(dataa.keys())))
            # Check if there is more than one recorded neuron in that area.
            if popVectors.shape[1]>1:
                template_left_pop = np.nanmean(popVectors[stimDiffs < 0, :], axis=0)
                template_right_pop = np.nanmean(popVectors[stimDiffs > 0, :], axis=0)

                if doFano:
                    fano = np.nan*np.ones(popVectors.shape[1])
                    for i in range(popVectors.shape[1]):
                        fano[i] = np.nanvar(popVectors[:,i])/np.nanmean(popVectors[:,i])
               # Loop over trials.
                for i in range(popVectors.shape[0]):
                    if stimDiffs[i] < 0:
                        cor_pop[i, bb] = np.corrcoef(template_left_pop, popVectors[i, :])[0, 1]
                        cor_wrong_pop[i, bb] = np.corrcoef(template_right_pop, popVectors[i, :])[0, 1]

                        if doSubSam:
                            # Loop over subsampling sizes
                            for k in range(numSamp):
                                # Loop over number of subsampling realizations.
                                for s in range(numSubRep):
                                        subsam = np.random.choice(popVectors.shape[1],int(np.floor(popVectors.shape[1]/(k+1))), replace = False)
                                        if np.sum(popVectors[i, subsam]) > 0:
                                            template_left_subsam = template_left_pop[subsam]
                                            template_right_subsam = template_right_pop[subsam]
                                            cor_subsam[i,bb,s,k] = np.corrcoef(template_left_subsam, popVectors[i, subsam])[0, 1]
                                            cor_wrong_subsam[i,bb,s,k] = np.corrcoef(template_right_subsam, popVectors[i, subsam])[0, 1]
                                        else:
                                            continue

                        # Shuffling loop.
                        if doShuff:
                            for s in range(numShuf):
                                aa = choices(np.arange(0, popVectors.shape[1],1), template_left_pop/np.sum(template_left_pop),
                                             k=int(np.sum(popVectors[i, :])))
                                rands = Counter(aa)
                                ww = np.zeros_like(popVectors[i, :])
                                ww[list(rands.keys())] = list(rands.values())
                                cor_Sh[i, bb, s] = \
                                np.corrcoef(ww, template_left_pop)[0, 1]
                                aaa = choices(np.arange(0, popVectors.shape[1],1), template_right_pop/np.sum(template_right_pop),
                                             k=int(np.sum(popVectors[i, :])))

                                rands = Counter(aaa)
                                ww = np.zeros_like(popVectors[i, :])
                                ww[list(rands.keys())] = list(rands.values())
                                cor_wrong_Sh[i, bb, s] = \
                                np.corrcoef(ww, template_right_pop)[0, 1]
                                specIdx_Sh[i,bb,s] =  cor_Sh[i,bb,s] - cor_wrong_Sh[i, bb, s]

                            perc[i,bb] = sts.percentileofscore(cor_Sh[i, bb, :], cor_pop[i, bb]) / 100
                            perc_wrong[i,bb] = sts.percentileofscore(cor_wrong_Sh[i, bb, :], cor_wrong_pop[i, bb]) / 100

                    elif stimDiffs[i] > 0:
                        cor_pop[i, bb] = np.corrcoef(template_right_pop, popVectors[i, :])[0, 1]
                        cor_wrong_pop[i, bb] = np.corrcoef(template_left_pop, popVectors[i, :])[0, 1]
                        if doSubSam:
                            # Loop over subsampling sizes
                            for k in range(numSamp):
                                # Loop over number of subsampling realizations.
                                for s in range(numSubRep):
                                    subsam = np.random.choice(popVectors.shape[1],int(np.floor(popVectors.shape[1]/(k+1))), replace = False)
                                    if np.sum(popVectors[i, subsam]) > 0:
                                        template_left_subsam = template_left_pop[subsam]
                                        template_right_subsam = template_right_pop[subsam]
                                        cor_subsam[i,bb,s,k] = np.corrcoef(template_right_subsam, popVectors[i, subsam])[0, 1]
                                        cor_wrong_subsam[i,bb,s,k] = np.corrcoef(template_left_subsam, popVectors[i, subsam])[0, 1]
                                    else:
                                        continue
                        # Shuffling loop.
                        if doShuff:
                            for s in range(numShuf):
                                aa = choices(np.arange(0, popVectors.shape[1], 1),
                                              template_right_pop / np.sum(template_right_pop),
                                              k=int(np.sum(popVectors[i, :])))
                                rands = Counter(aa)
                                ww = np.zeros_like(popVectors[i, :])
                                ww[list(rands.keys())] = list(rands.values())
                                cor_Sh[i, bb, s] = \
                                    np.corrcoef(ww, template_right_pop)[0, 1]

                                aaa = choices(np.arange(0, popVectors.shape[1]), template_left_pop/np.sum(template_left_pop),
                                             k=int(np.sum(popVectors[i, :])))
                                rands = Counter(aaa)
                                ww = np.zeros_like(popVectors[i, :])
                                ww[list(rands.keys())] = list(rands.values())
                                cor_wrong_Sh[i, bb, s] = np.corrcoef(ww, template_left_pop)[0, 1]
                                specIdx_Sh[i, bb, s] = cor_Sh[i, bb, s] - cor_wrong_Sh[i, bb, s]

                            perc[i,bb] = sts.percentileofscore(cor_Sh[i, bb, :], cor_pop[i, bb]) / 100
                            perc_wrong[i,bb] = sts.percentileofscore(cor_wrong_Sh[i, bb, :], cor_wrong_pop[i, bb]) / 100

                if doRemov:
                    # Neural removal, effect size computations
                    cor_remov_tmp_tmp = []
                    cor_remov_hit_tmp_tmp = []
                    cor_remov_miss_tmp_tmp = []
                    specIdx_remov_tmp_tmp = []
                    specIdx_remov_hit_tmp_tmp = []
                    specIdx_remov_miss_tmp_tmp = []
                    # One-neuron removal loop. Repeat all computations but removing a single neuron each time.
                    for g in range(num_cells):
                        pop2 = np.delete(popVectors.copy(), g, 1)
                        template_left = np.nanmean(pop2[stimDiffs < 0, :], axis=0)
                        template_right = np.nanmean(pop2[stimDiffs > 0, :], axis=0)
                        for i in range(pop2.shape[0]):
                            if stimDiffs[i] < 0:
                                cor[i,bb] = np.corrcoef(template_left, pop2[i, :])[0, 1]
                                cor_wrong[i,bb] = np.corrcoef(template_right, pop2[i, :])[0, 1]

                            elif stimDiffs[i] > 0:
                                cor[i,bb] = np.corrcoef(template_right, pop2[i, :])[0, 1]
                                cor_wrong[i,bb] = np.corrcoef(template_left, pop2[i, :])[0, 1]

                            elif stimDiffs[i] == 0:
                                cor[i,bb] = np.nan

                        spec_hit_pop = cor_pop[choicess==1,bb].ravel()-cor_wrong_pop[choicess==1,bb].ravel()
                        spec_miss_pop = cor_pop[choicess==-1,bb].ravel()-cor_wrong_pop[choicess==-1,bb].ravel()
                        spec_pop = cor_pop[:,bb].ravel()-cor_wrong_pop[:,bb].ravel()

                        cor_remov_tmp_tmp.append((cor_pop[:,bb] - cor[:,bb])/num_cells)
                        cor_remov_hit_tmp_tmp.append((cor_pop[choicess==1,bb] - cor[choicess==1,bb])/num_cells)
                        cor_remov_miss_tmp_tmp.append((cor_pop[choicess==-1,bb] - cor[choicess==-1,bb])/num_cells)
                        specIdx_remov_tmp_tmp.append((spec_pop - (cor[:,bb].ravel()-cor_wrong[:,bb].ravel()))/num_cells)
                        specIdx_remov_hit_tmp_tmp.append((spec_hit_pop - (cor[choicess==1,bb].ravel()-cor_wrong[choicess==1,bb].ravel()))/num_cells)
                        specIdx_remov_miss_tmp_tmp.append((spec_miss_pop - (cor[choicess==-1,bb].ravel()-cor_wrong[choicess==-1,bb].ravel()))/num_cells)
                    cor_remov_tmp.append(cor_remov_tmp_tmp)
                    cor_remov_hit_tmp.append(cor_remov_hit_tmp_tmp)
                    cor_remov_miss_tmp.append(cor_remov_miss_tmp_tmp)
                    specIdx_remov_tmp.append(specIdx_remov_hit_tmp_tmp)
                    specIdx_remov_hit_tmp.append(specIdx_remov_hit_tmp_tmp)
                    specIdx_remov_miss_tmp.append(specIdx_remov_miss_tmp_tmp)


    measured_corr.append(cor_pop)
    measured_corr_wrong.append(cor_wrong_pop)
    spec_idx.append(cor_pop - cor_wrong_pop)
    specIdx_hit.append(cor_pop[choicess == 1, :] - cor_wrong_pop[choicess == 1, :])
    specIdx_miss.append(cor_pop[choicess == -1, :] - cor_wrong_pop[choicess == -1, :])
    corr_hit.append(cor_pop[choicess == 1, :])
    corr_miss.append(cor_pop[choicess == -1, :])

    if doShuff:
        surrogate_corr.append(cor_Sh)
        surrogate_wrong_corr.append(cor_wrong_Sh)
        surrogate_specIdx.append(specIdx_Sh)
        perc_list.append(perc)
        perc_wrong_list.append(perc_wrong)

    if doSubSam:
        measured_corr_subsam.append(cor_subsam)
        measured_corr_subsam_wrong.append(cor_wrong_subsam)
        spec_idx_subsam.append(cor_subsam-cor_wrong_subsam)
        specIdx_hit_subsam.append(cor_subsam[choicess==1,:,:,:]-cor_wrong_subsam[choicess==1,:,:,:])
        specIdx_miss_subsam.append(cor_subsam[choicess==-1,:,:,:]-cor_wrong_subsam[choicess==-1,:,:,:])
        corr_hit_subsam.append(cor_subsam[choicess==1,:,:,:])
        corr_miss_subsam.append(cor_subsam[choicess==-1,:,:,:])
    if doRemov:
        cor_remov.append(cor_remov_tmp)
        cor_remov_hit.append(cor_remov_hit_tmp)
        cor_remov_miss.append(cor_remov_miss_tmp)
        specIdx_remov.append(specIdx_remov_tmp)
        specIdx_remov_hit.append(specIdx_remov_hit_tmp)
        specIdx_remov_miss.append(specIdx_remov_miss_tmp)
    print('m = ',m)

# Reorganize all quantities: pool together over sessions, for all areas.
flat_b = [item for sublist in b_uni for item in sublist]
uni_flat = np.unique(flat_b)
surrogate_fin = []
surrogate_wrong_fin = []
surrogate_specIdx_fin = []
measured_fin = []
measured_wrong_fin = []

perc_fin = []
perc_wrong_fin = []
spec_idx_fin = []
corr_hit_fin = []
corr_miss_fin = []
fano_fin = []
specIdx_hit_fin = []
specIdx_miss_fin = []

corr_remov_hit_fin = []
corr_remov_miss_fin = []
specIdx_remov_hit_fin = []
specIdx_remov_miss_fin = []

for bb in range(len(uni_flat)):

    subsam_tmp = []
    subsam_wrong_tmp = []

    meas_tmp = []
    measured_wrong_tmp = []
    perc_tmp = []
    perc_wrong_tmp = []
    surr_tmp = []
    surr_wrong_tmp = []
    specIdx_Sh_tmp = []
    spec_idx_tmp = []
    corr_hit_tmp = []
    corr_miss_tmp = []
    specIdx_hit_tmp = []
    specIdx_miss_tmp = []

    corr_remov_hit_tmp = []
    corr_remov_miss_tmp = []
    specIdx_remov_hit_tmp = []
    specIdx_remov_miss_tmp = []

    fano_tmp = []
    for m in range(39):
        if np.logical_and(uni_flat[bb] in b_uni[m],uni_flat[bb]!='root'):
            idx = np.where(b_uni[m] == uni_flat[bb])[0]
            if doFano:
                fano_tmp.append(fanoo[m][idx[0]])
            if doShuff:
                perc_tmp.append(perc_list[m][:,idx[0]])
                perc_wrong_tmp.append(perc_wrong_list[m][:,idx[0]])
                surr_tmp.append(surrogate_corr[m][:,idx[0],:].ravel())
                specIdx_Sh_tmp.append(surrogate_specIdx[m][:,idx[0],:].ravel())
                surr_wrong_tmp.append(surrogate_wrong_corr[m][:,idx[0]].ravel())
            if doRemov:
                corr_remov_hit_tmp.append([item for sublist in cor_remov_hit[m][idx[0]] for item in sublist])
                corr_remov_miss_tmp.append([item for sublist in cor_remov_miss[m][idx[0]] for item in sublist])
                specIdx_remov_hit_tmp.append([item for sublist in specIdx_remov_hit[m][idx[0]] for item in sublist])
                specIdx_remov_miss_tmp.append([item for sublist in specIdx_remov_miss[m][idx[0]] for item in sublist])

            meas_tmp.append(measured_corr[m][:,idx[0]])
            measured_wrong_tmp.append(measured_corr_wrong[m][:,idx[0]])
            corr_hit_tmp.append(corr_hit[m][:,idx[0]])
            corr_miss_tmp.append(corr_miss[m][:,idx[0]])
            spec_idx_tmp.append(spec_idx[m][:,idx[0]])
            specIdx_hit_tmp.append(specIdx_hit[m][:,idx[0]])
            specIdx_miss_tmp.append(specIdx_miss[m][:,idx[0]])
            spec_idx_tmp.append(spec_idx[m][:,idx[0]])

    measured_fin.append([item for sublist in meas_tmp for item in sublist])
    measured_wrong_fin.append([item for sublist in measured_wrong_tmp for item in sublist])
    spec_idx_fin.append([item for sublist in spec_idx_tmp for item in sublist])
    specIdx_hit_fin.append([item for sublist in specIdx_hit_tmp for item in sublist])
    specIdx_miss_fin.append([item for sublist in specIdx_miss_tmp for item in sublist])
    corr_hit_fin.append([item for sublist in corr_hit_tmp for item in sublist])
    corr_miss_fin.append([item for sublist in corr_miss_tmp for item in sublist])

    fano_fin.append([item for sublist in fano_tmp for item in sublist])

    surrogate_fin.append([item for sublist in surr_tmp for item in sublist])
    surrogate_specIdx_fin.append([item for sublist in specIdx_Sh_tmp for item in sublist])
    surrogate_wrong_fin.append([item for sublist in surr_wrong_tmp for item in sublist])
    perc_fin.append([item for sublist in perc_tmp for item in sublist])
    perc_wrong_fin.append([item for sublist in perc_wrong_tmp for item in sublist])

    corr_remov_hit_fin.append([item for sublist in corr_remov_hit_tmp for item in sublist])
    corr_remov_miss_fin.append([item for sublist in corr_remov_miss_tmp for item in sublist])
    specIdx_remov_hit_fin.append([item for sublist in specIdx_remov_hit_tmp for item in sublist])
    specIdx_remov_miss_fin.append([item for sublist in specIdx_remov_miss_tmp for item in sublist])


#Same, but for the subsampling loops:
if doSubSam:
    measured_subsam_fin = []
    measured_subsam_wrong_fin = []
    spec_idx_subsam_fin = []
    corr_hit_subsam_fin = []
    corr_miss_subsam_fin = []
    specIdx_hit_subsam_fin = []
    specIdx_miss_subsam_fin = []
    for bb in range(len(uni_flat)):
        subsam_tmp = []
        subsam_wrong_tmp = []
        spec_idx_subsam_tmp = []
        corr_hit_subsam_tmp = []
        corr_miss_subsam_tmp = []
        specIdx_hit_subsam_tmp = []
        specIdx_miss_subsam_tmp = []
        for k in range(numSamp):
            subsam_tmp_tmp = []
            subsam_wrong_tmp_tmp = []
            spec_idx_subsam_tmp_tmp = []
            corr_hit_subsam_tmp_tmp = []
            corr_miss_subsam_tmp_tmp = []
            specIdx_hit_subsam_tmp_tmp = []
            specIdx_miss_subsam_tmp_tmp = []
            for m in range(39):
                if np.logical_and(uni_flat[bb] in b_uni[m],uni_flat[bb]!='root'):
                    idx = np.where(b_uni[m] == uni_flat[bb])[0]
                    subsam_tmp_tmp.append(measured_corr_subsam[m][:,idx[0], :, k].ravel())
                    subsam_wrong_tmp_tmp.append(measured_corr_subsam_wrong[m][:,idx[0], :, k].ravel())
                    corr_hit_subsam_tmp_tmp.append(corr_hit_subsam[m][:,idx[0], :, k].ravel())
                    corr_miss_subsam_tmp_tmp.append(corr_miss_subsam[m][:,idx[0], :, k].ravel())
                    spec_idx_subsam_tmp_tmp.append(spec_idx_subsam[m][:,idx[0], :, k].ravel())
                    specIdx_hit_subsam_tmp_tmp.append(specIdx_hit_subsam[m][:,idx[0], :, k].ravel())
                    specIdx_miss_subsam_tmp_tmp.append(specIdx_miss_subsam[m][:,idx[0], :, k].ravel())

            subsam_tmp.append(subsam_tmp_tmp)
            subsam_wrong_tmp.append(subsam_wrong_tmp_tmp)
            corr_hit_subsam_tmp.append(corr_hit_subsam_tmp_tmp)
            corr_miss_subsam_tmp.append(corr_miss_subsam_tmp_tmp)
            spec_idx_subsam_tmp.append(spec_idx_subsam_tmp_tmp)
            specIdx_hit_subsam_tmp.append(specIdx_hit_subsam_tmp_tmp)
            specIdx_miss_subsam_tmp.append(specIdx_miss_subsam_tmp_tmp)
        # Area, subsample step, no. of sessions in which it was recorded, no. of repetitions x no. trials
        measured_subsam_fin.append(subsam_tmp)
        measured_subsam_wrong_fin.append(subsam_wrong_tmp)
        spec_idx_subsam_fin.append(spec_idx_subsam_tmp)
        specIdx_hit_subsam_fin.append(specIdx_hit_subsam_tmp)
        specIdx_miss_subsam_fin.append(specIdx_miss_subsam_tmp)
        corr_hit_subsam_fin.append(corr_hit_subsam_tmp)
        corr_miss_subsam_fin.append(corr_miss_subsam_tmp)



# Convert all lists into DataFrames, with the appropriate column-labelling (71 areas at the end).
df_measured = pd.DataFrame(measured_fin[:-1]).T
df_measured.columns = uni_flat[:-1]
df_measured.drop('OT',axis = 1,inplace=True)
df_measured_wrong = pd.DataFrame(measured_wrong_fin[:-1]).T
df_measured_wrong.columns = uni_flat[:-1]
df_measured_wrong.drop('OT',axis = 1,inplace=True)
df_specIdx_hit = pd.DataFrame(specIdx_hit_fin[:-1]).T
df_specIdx_hit.columns = uni_flat[:-1]
df_specIdx_hit.drop('OT',axis = 1,inplace=True)
df_specIdx_miss = pd.DataFrame(specIdx_miss_fin[:-1]).T
df_specIdx_miss.columns = uni_flat[:-1]
df_specIdx_miss.drop('OT',axis = 1,inplace=True)
df_corr_hit = pd.DataFrame(corr_hit_fin[:-1]).T
df_corr_hit.columns = uni_flat[:-1]
df_corr_hit.drop('OT',axis = 1,inplace=True)
df_corr_miss = pd.DataFrame(corr_miss_fin[:-1]).T
df_corr_miss.columns = uni_flat[:-1]
df_corr_miss.drop('OT',axis = 1,inplace=True)
df_spec_idx = pd.DataFrame(spec_idx_fin[:-1]).T
df_spec_idx.columns = uni_flat[:-1]
df_spec_idx.drop('OT',axis = 1,inplace=True)

if doShuff:
    df_surr = pd.DataFrame(surrogate_fin[:-1]).T
    df_surr.columns = uni_flat[:-1]
    df_surr.drop('OT',axis = 1,inplace=True)
    df_surr_specIdx = pd.DataFrame(surrogate_specIdx_fin[:-1]).T
    df_surr_specIdx.columns = uni_flat[:-1]
    df_surr_specIdx.drop('OT',axis = 1,inplace=True)
    df_surr_wrong = pd.DataFrame(surrogate_wrong_fin[:-1]).T
    df_surr_wrong.columns = uni_flat[:-1]
    df_surr_wrong.drop('OT',axis = 1,inplace=True)
    df_perc = pd.DataFrame(perc_fin[:-1]).T
    df_perc.columns = uni_flat[:-1]
    df_perc.drop('OT',axis = 1,inplace=True)
    df_perc_wrong = pd.DataFrame(perc_wrong_fin[:-1]).T
    df_perc_wrong.columns = uni_flat[:-1]
    df_perc_wrong.drop('OT',axis = 1,inplace=True)

if doFano:
    df_fano = pd.DataFrame(fano_fin[:-1]).T
    df_fano.columns = uni_flat[:-1]
    df_fano.drop('OT', axis=1, inplace=True)

if doRemov:
    df_corr_remov_hit = pd.DataFrame(corr_remov_hit_fin[:-1]).T
    df_corr_remov_hit.columns = uni_flat[:-1]
    df_corr_remov_hit.drop('OT',axis = 1,inplace=True)
    df_corr_remov_miss = pd.DataFrame(corr_remov_miss_fin[:-1]).T
    df_corr_remov_miss.columns = uni_flat[:-1]
    df_corr_remov_miss.drop('OT',axis = 1,inplace=True)

    df_specIdx_remov_hit = pd.DataFrame(specIdx_remov_hit_fin[:-1]).T
    df_specIdx_remov_hit.columns = uni_flat[:-1]
    df_specIdx_remov_hit.drop('OT',axis = 1,inplace=True)
    df_specIdx_remov_miss = pd.DataFrame(specIdx_remov_miss_fin[:-1]).T
    df_specIdx_remov_miss.columns = uni_flat[:-1]
    df_specIdx_remov_miss.drop('OT',axis = 1,inplace=True)


# Load the choice and stimulus information quantities for each area over sessions (from the decoder analyses).
df_choice = pd.read_csv('df_test_choiceInfo_allResponsive_200ms.csv')
df_choice.drop('Unnamed: 0', axis = 1, inplace = True)
df_stim = pd.read_csv('df_test_stimInfo_allResponsive_200ms.csv')
df_stim.drop('Unnamed: 0', axis = 1, inplace = True)

# From the experimental datapoints, we construct the corresponding continuous distribution via Gaussian Kernel Estimation.
density_stim = sts.kde.gaussian_kde(df_stim.median())
density_stim = density_stim(np.unique(df_stim.median()))

density_choi = sts.kde.gaussian_kde(df_choice.median())
density_choi = density_choi(np.unique(df_choice.median()))

# Using the elbow method to select the stimulus-, choice- or both-informative areas.
knee_stim = kn.KneeLocator(np.unique(df_stim.median())[np.where(density_stim==np.max(density_stim))[0][0]+10:],
                      density_stim[np.where(density_stim==np.max(density_stim))[0][0]+10:],
                                      curve='convex', direction='decreasing', interp_method='interp1d')

knee_choi = kn.KneeLocator(np.unique(df_choice.median())[np.where(density_choi==np.max(density_choi))[0][0]+10:],
                      density_choi[np.where(density_choi==np.max(density_choi))[0][0]+10:],
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

# Statistical comparisons over all areas (splitting by related and unrelated ones)
median_correct = df_measured.median()
median_incorrect = df_measured_wrong.median()
median_diffs = median_correct - median_incorrect

median_diffs_related = median_diffs[np.logical_and(df_rel.values[0] != 1, df_rel.values[0] != 'u_Unrelated')]
prod_info = df_stim.median()*df_choice.median()
ar = np.argsort(prod_info)
prod_sorted = prod_info.iloc[ar.values]
median_diffs_unrelated_bot = prod_sorted[:9]
median_diffs_unrelated_top = prod_sorted[-19:-10]
sts.ttest_ind(median_diffs_related, prod_sorted[9:], equal_var= False)


# We will now just work with those areas that are task-related and the 3 other examples (ACB, SI and EPd).
selected = df_rel.values!=1
df_rel_fin = df_rel.iloc[:,selected[0]]
aa = np.argsort(df_rel_fin)
b = df_rel_fin.iloc[:,aa.values[0]]
b_vis = b.reindex(sorted(b.columns[b.values[0] == 'a_Visual']), axis=1)
b_both = b.reindex(sorted(b.columns[b.values[0] == 'b_Both']), axis=1)
b_choi = b.reindex(sorted(b.columns[b.values[0] == 'c_Choice']), axis=1)
b_unrel = b.reindex(sorted(b.columns[b.values[0] == 'u_Unrelated']), axis=1)
b_fin = pd.concat([b_vis,b_both,b_choi, b_unrel], axis = 1)

# Out of all of the recorded regions and subregions, we just extract the selected ones, for all quantities.
# We then relabel the columns of all DataFrames, for better inspection and plotting purposes.

if doFano:
    df_fano_fin = df_fano.iloc[:,selected[0]]
    df_fano_fin2 = df_fano_fin[b_fin.columns]

df_measured_fin = df_measured.iloc[:,selected[0]]
df_wrong_fin = df_measured_wrong.iloc[:,selected[0]]
df_hit_fin = df_corr_hit.iloc[:,selected[0]]
df_miss_fin = df_corr_miss.iloc[:,selected[0]]
df_spec_idx_fin = df_spec_idx.iloc[:,selected[0]]
df_specIdx_hit_fin = df_specIdx_hit.iloc[:,selected[0]]
df_specIdx_miss_fin = df_specIdx_miss.iloc[:,selected[0]]

df_measured_fin2 = df_measured_fin[b_fin.columns]
df_wrong_fin2 = df_wrong_fin[b_fin.columns]
df_hit_fin2 = df_hit_fin[b_fin.columns]
df_miss_fin2 = df_miss_fin[b_fin.columns]
df_spec_idx_fin2 = df_spec_idx_fin[b_fin.columns]
df_specIdx_hit_fin2 = df_specIdx_hit_fin[b_fin.columns]
df_specIdx_miss_fin2 = df_specIdx_miss_fin[b_fin.columns]

if doRemov:
    df_corr_remov_hit_fin = df_corr_remov_hit.iloc[:,selected[0]]
    df_corr_remov_miss_fin = df_corr_remov_miss.iloc[:,selected[0]]
    df_specIdx_remov_hit_fin = df_specIdx_remov_hit.iloc[:,selected[0]]
    df_specIdx_remov_miss_fin = df_specIdx_remov_miss.iloc[:,selected[0]]

    df_corr_remov_hit_fin2 = df_corr_remov_hit_fin[b_fin.columns]
    df_corr_remov_miss_fin2 = df_corr_remov_miss_fin[b_fin.columns]
    df_specIdx_remov_hit_fin2 = df_specIdx_remov_hit_fin[b_fin.columns]
    df_specIdx_remov_miss_fin2 = df_specIdx_remov_miss_fin[b_fin.columns]

if doShuff:
    df_perc_fin = df_perc.iloc[:,selected[0]]
    df_perc_wrong_fin = df_perc_wrong.iloc[:,selected[0]]
    df_surr_fin = df_surr.iloc[:,selected[0]]
    df_surr_specIdx_fin = df_surr_specIdx.iloc[:,selected[0]]
    df_surr_wrong_fin = df_surr_wrong.iloc[:,selected[0]]

    df_perc_fin2 = df_perc_fin[b_fin.columns]
    df_perc_wrong_fin2 = df_perc_wrong_fin[b_fin.columns]
    df_surr_fin2 = df_surr_fin[b_fin.columns]
    df_surr_specIdx_fin2 = df_surr_specIdx_fin[b_fin.columns]
    df_surr_wrong_fin2 = df_surr_wrong_fin[b_fin.columns]


# U-tests between different quantities (change u and v).
sign_2 = []
pval_2 = []
v1 = []
v2 = []
for i in range(df_measured_fin2.shape[1]):
    u = df_measured_fin2.iloc[:,i]
    v = df_wrong_fin2.iloc[:,i]
    s, p = sts.mannwhitneyu(u, v, alternative = 'greater')
    sign_2.append(s)
    pval_2.append(p)
print(sign_2)
print(pval_2)

# If there are multiple comparisons to be corrected for, use a FDR Benjamini-Hochberg procedure, with a FWER of 0.05:
doMulti = 0
if doMulti:
    from statsmodels.stats.multitest import multipletests as mt
    mt(np.array(pval_2), alpha = 0.05, method = 'fdr_bh')


## -------------------- Plotting part. --------------------

# Color code that we use throughout the paper: stim-informative areas in shades of blue, choice-informative in red,
# both-informative in purple and uninformative in gray.
colors = [sns.desaturate('blue', 0.2), sns.desaturate('red', 0.3), sns.desaturate('purple', 0.2), sns.desaturate('gray', 0.3)]
colors2 = [sns.desaturate('blue', 0.9), sns.desaturate('red', 0.7), sns.desaturate('purple', 0.9), sns.desaturate('black', 0.5)]

if doShuff:
    fig, ax = plt.subplots(nrows=1)
    plotBoxes(ax,df_spec_idx_fin2, df_surr_specIdx_fin2,b_fin,colors, colors2)
    plt.axhline(0, color= 'black', linewidth = 2, alpha = 0.5, zorder = 1)

data1 = df_measured_fin2
data2 = df_wrong_fin2
fig, ax = plt.subplots(figsize=(12, 8))
plotViolin(ax, data1, data2, b_fin, colors, colors2, 'CorrWrong')

data1 = df_hit_fin2
data2 = df_miss_fin2
fig, ax = plt.subplots(figsize=(12, 8))
plotViolin(ax, data1, data2, b_fin, colors, colors2, 'HitMiss')


fig, ax = plt.subplots(figsize=(8, 1))
plotEffSizes(ax, data1, data2, b_fin, colors, colors2)


data1 = df_specIdx_remov_hit_fin2
data2 = df_specIdx_remov_miss_fin2
fig, ax = plt.subplots()
# plotScatterRemoval(ax, data1, data2, b_fin, colors, colors2, 'Correlation')
plotScatterRemoval(ax, data1, data2, b_fin, colors, colors2, 'Spec Idx')

greater1 = []
greater2 = []
for i in range(data1.shape[1]):
    a = np.sum(data1.iloc[:, i]>=0)/np.sum(~np.isnan(data1.iloc[:,i]))*100
    a2 = np.sum(data2.iloc[:, i]>=0)/np.sum(~np.isnan(data2.iloc[:,i]))*100
    greater1.append(a)
    greater2.append(a2)

plt.figure(figsize=(8,1))
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
    plt.plot(i, greater1[i], marker = 'o', alpha = 0.5, color = color2, markerfacecolor = color, markersize = 6)
    plt.plot(i+0.3, greater2[i], marker = 'o', alpha = 0.5, color = color, markerfacecolor = color2, markersize = 6)
    plt.plot([i,i+0.3],[greater1[i],greater2[i]],color = color)

plt.xticks(ticks = np.arange(0.2,data1.shape[1]+0.125,1), labels = data1.columns, rotation = 30, fontsize = 16)
plt.axhline(50, lw = 2, ls = '--', color = 'gray', alpha = 0.5)
plt.ylabel('% of positive points')
plt.ylim([0, 100])

#Subsampling loops and plotting part:
if doSubSam:
    measured_subsam_fin2 = []
    measured_subsam_wrong_fin2 = []
    spec_idx_subsam_fin2 = []
    corr_hit_subsam_fin2 = []
    corr_miss_subsam_fin2 = []
    specIdx_hit_subsam_fin2 = []
    specIdx_miss_subsam_fin2 = []
    for bb in range(len(uni_flat)):
        if uni_flat[bb] in b_fin.columns:
            subsam_tmp = []
            subsam_wrong_tmp = []
            spec_idx_subsam_tmp = []
            corr_hit_subsam_tmp = []
            corr_miss_subsam_tmp = []
            specIdx_hit_subsam_tmp = []
            specIdx_miss_subsam_tmp = []
            for k in range(numSamp):
                subsam_tmp.append([item for sublist in measured_subsam_fin[bb][k] for item in sublist])
                subsam_wrong_tmp.append([item for sublist in measured_subsam_wrong_fin[bb][k] for item in sublist])
                corr_hit_subsam_tmp.append([item for sublist in corr_hit_subsam_fin[bb][k] for item in sublist])
                corr_miss_subsam_tmp.append([item for sublist in corr_miss_subsam_fin[bb][k] for item in sublist])
                spec_idx_subsam_tmp.append([item for sublist in spec_idx_subsam_fin[bb][k] for item in sublist])
                specIdx_hit_subsam_tmp.append([item for sublist in specIdx_hit_subsam_fin[bb][k] for item in sublist])
                specIdx_miss_subsam_tmp.append([item for sublist in specIdx_miss_subsam_fin[bb][k] for item in sublist])

            measured_subsam_fin2.append(subsam_tmp)
            measured_subsam_wrong_fin2.append(subsam_wrong_tmp)
            spec_idx_subsam_fin2.append(spec_idx_subsam_tmp)
            specIdx_hit_subsam_fin2.append(specIdx_hit_subsam_tmp)
            specIdx_miss_subsam_fin2.append(specIdx_miss_subsam_tmp)
            corr_hit_subsam_fin2.append(corr_hit_subsam_tmp)
            corr_miss_subsam_fin2.append(corr_miss_subsam_tmp)

    # If you computed the subsampling quantities, this part plots the layout that can be found in the paper (Fig. 5).
    v_list = [measured_subsam_fin2, measured_subsam_wrong_fin2, corr_hit_subsam_fin2, corr_miss_subsam_fin2, specIdx_hit_subsam_fin2, specIdx_miss_subsam_fin2]

    fig, axes = plt.subplots(nrows=4, ncols=12)
    plotSubsam(axes, v_list, spec_idx_subsam_fin2, b_fin, colors, colors2, numSamp)
