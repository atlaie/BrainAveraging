# We begin by importing all necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
from matplotlib.cm import register_cmap
import matplotlib.patches as pch
import kneed as kn

# Setting up plotting options and define a custom palette.
rgb = [(1,1,1), ( 73/255, 111/255, 252/255 ), ( 252/255, 102/255, 65/255 )]
custom2 = sns.blend_palette(rgb, n_colors=255, as_cmap=True)
register_cmap("customBlend", custom2)
font = {'family' : 'normal',
        'size'   : 26}
plt.rc('font', **font)
sns.set_style('white')
plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
plt.rc('text', usetex=False)
np.seterr(divide='ignore', invalid='ignore')

# We define a custom function for the Mutual Information calculation. Alternatively, one could use the one
# implemented in sklearn.metrics
def computeMI(x, y):
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([len(x[x == xval]) / float(len(x)) for xval in x_value_list])  # P(x)
    Py = np.array([len(y[y == yval]) / float(len(y)) for yval in y_value_list])  # P(y)
    for i in range(len(x_value_list)):
        if Px[i] == 0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy) == 0:
            continue
        pxy = np.array([len(sy[sy == yval]) / float(len(y)) for yval in y_value_list])  # p(x,y)
        t = pxy[Py > 0.] / Py[Py > 0.] / Px[i]  # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t > 0] * np.log2(t[t > 0]))  # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi

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
flat_b = [item for sublist in b_all for item in sublist]

# As all our results have to be cross-validated (CV), here we define the kind of CV we will implement. We use a
# repeated (for greater fidelity) stratified (as we have unbalanced data) 5-fold CV.
n_repeats = 5
n_splits = 5
skf = RepeatedStratifiedKFold(n_splits=n_splits, random_state=0, n_repeats=n_repeats)

# We will perform hyperparameter tuning, in this case over the L2-penalty strength. As we're not interested in sparsity,
# we don't use L1 nor elastic net penalties.
n_alphas = 20
alpha = np.logspace(-2,2,n_alphas)

# Preallocation of the different metrics we will save (both for the stimulus and the choice decoders). Even if we
# did not explicitly reported them in the paper, we did make sure that the decoder outperformed two null models:
# majority class (always predicting the most abundant label) and random predictions.
scores_rand = np.zeros((39,50))
scores_maj = np.zeros(39)
f1_test_choice = []
logL_test_choice = []
wAcc_test_choice = []
AUROC_test_choice = []
MutInfo_choice = []
MutInfo_choice_Sh = []

f1_test_stim = []
logL_test_stim = []
wAcc_test_stim = []
AUROC_test_stim = []
MutInfo_stim = []
MutInfo_stim_Sh = []
perc_stimulus = []

# Loop over experimental sessions (39).
for m in range(39):

    # We load the population vectors we computed in SpikesOverSessions.py
    filename = 'PopVectors_responsive200ms_session%d.npz' % m
    dataa = np.load(filename)

    # Our stimulus feature is the contrast difference between screens. We remove those trials in which there
    # was a projected image and the contrast difference was null.
    filename_stimDiffs = 'StimDifference_session%d.npy' % m
    stimDiffs = np.load(filename_stimDiffs)
    y_stim2 = LabelEncoder().fit_transform(stimDiffs.ravel())
    filename_stims = 'Stims_session%d.npz' % m
    stims = np.load(filename_stims)
    stims_left = stims['left']
    stims_right = stims['right']
    stims_left = stims_left.reshape(stims_left.shape[0], )
    stims_right = stims_right.reshape(stims_right.shape[0], )

    idx_filter = ~np.logical_and(stims_left == stims_right, stims_left != 0)
    y_stim = y_stim2[idx_filter]

    # We will also decode choices for those same trials.
    filename_choices = 'Choices_session%d.npy' % m
    choices = np.load(filename_choices)
    y_choice = LabelEncoder().fit_transform(choices.ravel())
    y_choice = y_choice[idx_filter]

    # Preallocation of the relevant metrics.
    f1_test_choice_tmp = np.zeros(len(dataa.keys()))
    logL_test_choice_tmp = np.zeros(len(dataa.keys()))
    wAcc_test_choice_tmp = np.zeros(len(dataa.keys()))
    MutInfo_choice_tmp = np.zeros(len(dataa.keys()))

    # Loop over recorded areas.
    for bb in range(len(dataa.keys())):
        num_cells = dataa[list(dataa.keys())[bb]].shape[1]
        # Check if there's more than one recorded neuron in that area and that all of the relevant shapes match.
        if np.logical_and(dataa[list(dataa.keys())[bb]][idx_filter,:].shape[0] == y_choice.shape[0],num_cells>1):
            # We split our data into 80% training and 20% test.
            X_train_choice, X_test_choice, y_train_choice, y_test_choice = train_test_split(dataa[list(dataa.keys())[bb]][idx_filter,:],
                                                                                            y_choice, stratify=y_choice, test_size=0.2, random_state=42)
            # The data has to be scaled, in order to be able to train the decoder. We do each scaling separately to avoid
            # the information leakage problem.
            scaler = MinMaxScaler()
            X_train_choice = scaler.fit_transform(X_train_choice)
            X_test_choice = scaler.transform(X_test_choice)
            if X_train_choice.shape[1]>0:
                # We perform data augmentation by duplicating the datapoints we have. This is done to increase
                # prediction power when we have a small number of observations (trials).
                X_train_Aug_choice = np.stack((X_train_choice, X_train_choice)).reshape(len(y_train_choice) * 2, X_train_choice.shape[1])
                y_train_Aug_choice = np.stack((y_train_choice, y_train_choice)).reshape(len(y_train_choice) * 2)
                # The decoder will be a L2-regularized Logistic regression with balanced weight classes (unnecessary
                # if we're also stratifying the cross-validation, in principle, but here just to make sure that we're
                # not using an imbalanced dataset at any random iteration of the CV).
                # We will minimize the logLoss (cross-entropy).
                clf = LogisticRegressionCV(Cs = alpha,solver='lbfgs', max_iter=500,
                                         class_weight='balanced', penalty='l2', cv = skf, refit=True, n_jobs=6, random_state=0,
                                           scoring = 'neg_log_loss', multi_class='auto')
                clf.fit(X_train_Aug_choice, y_train_Aug_choice)
                y_pred_choice = clf.predict(X_test_choice)
                yhat_choice = clf.predict_proba(X_test_choice)
                f1_test_choice_tmp[bb] = metrics.f1_score(y_test_choice, y_pred_choice, average = 'macro')
                logL_test_choice_tmp[bb] = metrics.log_loss(y_test_choice, yhat_choice)
                wAcc_test_choice_tmp[bb] = metrics.balanced_accuracy_score(y_test_choice, y_pred_choice)
                MutInfo_choice_tmp[bb] = computeMI(y_test_choice,y_pred_choice)
            else:
                f1_test_choice_tmp[bb] = np.nan
                logL_test_choice_tmp[bb] = np.nan
                wAcc_test_choice_tmp[bb] = np.nan
                MutInfo_choice_tmp[bb] = np.nan

    f1_test_choice.append(f1_test_choice_tmp)
    logL_test_choice.append(logL_test_choice_tmp)
    wAcc_test_choice.append(wAcc_test_choice_tmp)
    MutInfo_choice.append(MutInfo_choice_tmp)

    # Same, for the stimulus decoder.
    f1_test_stim_tmp = np.zeros(len(dataa.keys()))
    logL_test_stim_tmp = np.zeros(len(dataa.keys()))
    wAcc_test_stim_tmp = np.zeros(len(dataa.keys()))
    MutInfo_stim_tmp = np.zeros(len(dataa.keys()))
    for bb in range(len(dataa.keys())):
        num_cells = dataa[list(dataa.keys())[bb]].shape[1]
        if np.logical_and(dataa[list(dataa.keys())[bb]][idx_filter,:].shape[0] == y_stim.shape[0], num_cells > 1):
            X_train_stim, X_test_stim, y_train_stim, y_test_stim = train_test_split(dataa[list(dataa.keys())[bb]][idx_filter,:],
                                                                                    y_stim, stratify=y_stim,
                                                                                    test_size=0.2, random_state=42)
            scaler = MinMaxScaler()
            X_train_stim = scaler.fit_transform(X_train_stim)
            X_test_stim = scaler.transform(X_test_stim)
            if X_train_stim.shape[1] > 0:
                X_train_Aug_stim = np.stack((X_train_stim, X_train_stim)).reshape(len(y_train_stim) * 2,
                                                                                  X_train_stim.shape[1])
                y_train_Aug_stim = np.stack((y_train_stim, y_train_stim)).reshape(len(y_train_stim) * 2)
                clf = LogisticRegressionCV(Cs=alpha, solver='lbfgs', max_iter=500,
                                           class_weight='balanced', penalty='l2', cv=skf, refit=True, n_jobs=6,
                                           random_state=0,
                                           scoring='neg_log_loss', multi_class='auto')
                clf.fit(X_train_Aug_stim, y_train_Aug_stim)
                y_pred_stim = clf.predict(X_test_stim)
                yhat_stim = clf.predict_proba(X_test_stim)

                f1_test_stim_tmp[bb] = metrics.f1_score(y_test_stim, y_pred_stim, average='macro')
                logL_test_stim_tmp[bb] = metrics.log_loss(y_test_stim, yhat_stim)
                wAcc_test_stim_tmp[bb] = metrics.balanced_accuracy_score(y_test_stim, y_pred_stim)
                MutInfo_stim_tmp[bb] = computeMI(y_test_stim, y_pred_stim)
            else:
                f1_test_stim_tmp[bb] = np.nan
                logL_test_stim_tmp[bb] = np.nan
                wAcc_test_stim_tmp[bb] = np.nan
                MutInfo_stim_tmp[bb] = np.nan

    f1_test_stim.append(f1_test_stim_tmp)
    logL_test_stim.append(logL_test_stim_tmp)
    wAcc_test_stim.append(wAcc_test_stim_tmp)
    MutInfo_stim.append(MutInfo_stim_tmp)
    print(m)


# Reorganize all quantities: pool together over sessions, for all areas. We're just using MutInfos in this case,
# we could also save the other metrics if desired.
flat_b = [item for sublist in b_uni for item in sublist]
uni_flat = np.unique(flat_b)

scor_test_choice = []
scor_test_stim = []
for bb in range(len(uni_flat)):
    scor_test_choice_tmp = []
    scor_test_stim_tmp = []
    for m in range(39):
        if np.logical_and(uni_flat[bb] in b_uni[m], uni_flat[bb] != 'root'):
            idx = np.where(b_uni[m] == uni_flat[bb])[0]
            scor_test_choice_tmp.append(MutInfo_choice[m][idx])
            scor_test_stim_tmp.append(MutInfo_stim[m][idx])
    scor_test_choice.append([item for sublist in scor_test_choice_tmp for item in sublist])
    scor_test_stim.append([item for sublist in scor_test_stim_tmp for item in sublist])


# Convert all lists into DataFrames, with the appropriate column-labelling (71 areas at the end).
df_stim = pd.DataFrame(scor_test_stim[:-1]).T
df_stim.columns = uni_flat[:-1]
df_stim.drop('OT',axis = 1,inplace=True)
df_choice = pd.DataFrame(scor_test_choice[:-1]).T
df_choice.columns = uni_flat[:-1]
df_choice.drop('OT',axis = 1,inplace=True)

#As stated in the paper, each point is centered at the median over sessions and the circle radius is proportional
# to the std.
weights = np.nanmedian([df_stim.std(), df_choice.std()],axis = 0)*1000
weights[np.isnan(weights)] = 10

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
# For plotting purposes, we shift the area labels by an small number.
offset = 0.015

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


## -------------------- Plotting part. --------------------

# Color code that we use throughout the paper: stim-informative areas in shades of blue, choice-informative in red,
# both-informative in purple and uninformative in gray.

fig, ax = plt.subplots(figsize = (13,14))
sq = pch.Rectangle((inf_stim,0), np.max(df_stim.median().values)-inf_stim+offset, inf_choi, color='blue', alpha = 0.3,linewidth=0, zorder = 0)
sq2 = pch.Rectangle((0,inf_choi), inf_stim, np.max(df_choice.median().values)-inf_choi+offset, color='red', alpha = 0.3,linewidth=0, zorder = 0)
sq3 = pch.Rectangle((inf_stim,inf_choi), np.max(df_stim.median().values)-inf_stim+offset,
                    np.max(df_choice.median().values)-inf_choi+offset,color='purple', alpha = 0.5,linewidth=0, zorder = 0)
ax.add_patch(sq)
ax.add_patch(sq2)
ax.add_patch(sq3)
counter = 0
for col_id in df_choice.columns:
    if counter%2 == 0:
        ax.annotate(col_id, (df_stim[col_id].median(),df_choice[col_id].median()), fontsize = 12, zorder = 2)
    else:
        ax.annotate(col_id, (df_stim[col_id].median()+0.002,df_choice[col_id].median()+0.002), fontsize = 12, zorder = 2)
    plt.scatter(x=df_stim[col_id].median(), y=df_choice[col_id].median(), s=weights[counter],
                color=sns.desaturate('black', 0.3), alpha = 0.8, zorder = 1)
    counter += 1
plt.xlabel('Stimulus Information')
plt.ylabel('Choice Information')


fig, ax = plt.subplots(figsize=(13,3))
sns.histplot(df_stim.median(), bins = 40, kde = False, color = sns.desaturate('blue', 0.4))
plt.plot(np.unique(df_stim.median()),density_stim, color = sns.desaturate('blue', 0.4), lw = 2)
plt.axvline(inf_stim, color = sns.desaturate('blue', 0.4), lw = 2, ls = '--')

fig, ax = plt.subplots(figsize=(13,3))
plt.plot(np.unique(df_choice.median()),density_choi, color = sns.desaturate('red', 0.4), lw = 2)
sns.histplot(df_choice.median(), bins = 40, kde = False, color = sns.desaturate('red', 0.4))
plt.axvline(inf_choi, color = sns.desaturate('red', 0.4), lw = 2, ls = '--')
