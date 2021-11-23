import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import scipy.stats as sts

def plotViolin(ax, data1, data2, b_fin, colors, colors2, legend, plotSpec):
    for i in range(data1.shape[1]):
        d1 = data1.iloc[:,i]
        v1 = ax.violinplot(dataset = [d1[~np.isnan(d1)]], positions=[i],
                       showmeans=False, showextrema=False, showmedians=True)
        for b in v1['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
            b.set_color('b')
            if b_fin.iloc[:,i].values == 'a_Visual':
                b.set_color(colors[0])
                b.set_facecolor(colors[0])
            elif b_fin.iloc[:,i].values == 'c_Choice':
                b.set_color(colors[1])
                b.set_facecolor(colors[1])
            elif b_fin.iloc[:,i].values == 'b_Both':
                b.set_color(colors[2])
                b.set_facecolor(colors[2])
            elif b_fin.iloc[:, i].values == 'u_Unrelated':
                b.set_color(colors[3])
                b.set_facecolor(colors[3])

        b = v1['cmedians']
        if b_fin.iloc[:, i].values == 'a_Visual':
            b.set_color(colors[0])
        elif b_fin.iloc[:, i].values == 'c_Choice':
            b.set_color(colors[1])
        elif b_fin.iloc[:, i].values == 'b_Both':
            b.set_color(colors[2])
        elif b_fin.iloc[:, i].values == 'u_Unrelated':
            b.set_color(colors[3])

        d2 = data2.iloc[:,i]
        v2 = ax.violinplot(dataset = [d2[~np.isnan(d2)]], positions=[i],
                       showmeans=False, showextrema=False, showmedians=True)
        for b in v2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further left than the center
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)

            if b_fin.iloc[:, i].values == 'a_Visual':
                b.set_color(colors[0])
                b.set_facecolor(colors2[0])
            elif b_fin.iloc[:, i].values == 'c_Choice':
                b.set_color(colors[1])
                b.set_facecolor(colors2[1])
            elif b_fin.iloc[:, i].values == 'b_Both':
                b.set_color(colors[2])
                b.set_facecolor(colors2[2])
            elif b_fin.iloc[:, i].values == 'u_Unrelated':
                b.set_color(colors[3])
                b.set_facecolor(colors2[3])

        b = v2['cmedians']
        if b_fin.iloc[:, i].values == 'a_Visual':
            b.set_color(colors2[0])
        elif b_fin.iloc[:, i].values == 'c_Choice':
            b.set_color(colors2[1])
        elif b_fin.iloc[:, i].values == 'b_Both':
            b.set_color(colors2[2])
        elif b_fin.iloc[:, i].values == 'u_Unrelated':
            b.set_color(colors2[3])

    if plotSpec:
        plt.ylabel('Specificity Index', fontsize = 22)
    else:
        plt.ylabel('Correlation', fontsize = 22)
    plt.xticks(ticks = np.arange(0,data1.shape[1]), labels = data1.columns, rotation = 30, fontsize = 22)
    if legend == 'HitMiss':
        blue_patch = mpatches.Patch(color=colors[0], alpha = 0.5, label = 'Visual, hits')
        red_patch = mpatches.Patch(color=colors[1], alpha = 0.5, label='Choice, hits')
        purple_patch = mpatches.Patch(color=colors[2], alpha = 0.5, label = 'Both, hits')
        gray_patch = mpatches.Patch(color=colors[3], alpha = 0.5, label = 'Unrelated, hits')

        blue_patch_dark = mpatches.Patch(color=colors2[0], alpha = 0.5, label = 'Visual, misses')
        red_patch_dark = mpatches.Patch(color=colors2[1], alpha = 0.5, label='Choice, misses')
        purple_patch_dark = mpatches.Patch(color=colors2[2], alpha = 0.5, label = 'Both, misses')
        gray_patch_dark = mpatches.Patch(color=colors2[3], alpha = 0.5, label = 'Unrelated, misses')
        plt.legend(handles=[blue_patch, blue_patch_dark, red_patch, red_patch_dark, purple_patch, purple_patch_dark, gray_patch, gray_patch_dark],
                   fancybox=True, framealpha=1, shadow=True, loc='lower left', bbox_to_anchor=[0.2, 0.08, 0.5, 0.08],
                   bbox_transform=plt.gcf().transFigure, ncol=4, prop={'size': 22})
    if legend == 'CorrWrong':
        blue_patch = mpatches.Patch(color=colors[0], alpha = 0.5, label = 'Visual, correct template')
        red_patch = mpatches.Patch(color=colors[1], alpha = 0.5, label='Choice, correct template')
        purple_patch = mpatches.Patch(color=colors[2], alpha = 0.5, label = 'Both, correct template')
        gray_patch = mpatches.Patch(color=colors[3], alpha = 0.5, label = 'Unrelated, correct template')

        blue_patch_dark = mpatches.Patch(color=colors2[0], alpha = 0.5, label = 'Visual, incorrect template')
        red_patch_dark = mpatches.Patch(color=colors2[1], alpha = 0.5, label='Choice, incorrect template')
        purple_patch_dark = mpatches.Patch(color=colors2[2], alpha = 0.5, label = 'Both, incorrect template')
        gray_patch_dark = mpatches.Patch(color=colors2[3], alpha = 0.5, label = 'Unrelated, incorrect template')

        plt.legend(handles=[blue_patch, blue_patch_dark, red_patch, red_patch_dark, purple_patch, purple_patch_dark, gray_patch, gray_patch_dark],
                   fancybox=True, framealpha=1, shadow=True, loc='lower left', bbox_to_anchor=[0.05, 0.08, 0.35, 0.08],
                   bbox_transform=plt.gcf().transFigure, ncol=4, prop={'size': 22})
    plt.subplots_adjust(top=0.976,bottom=0.067,left=0.064,right=0.987,hspace=0.174,wspace=0.2)
    plt.xticks(ticks=np.arange(0, data1.shape[1]), labels=data1.columns, rotation=30, fontsize=26)


def plotBoxes(ax, data1, data1_surr, b_fin, colors, colors2,medians):

    cajas = sns.boxplot(data=data1, orient='v', color='blue',
                        boxprops=dict(color=colors[0], facecolor="blue", alpha=0.3),
                        medianprops=dict(color='blue', linewidth=3), whiskerprops=dict(color='blue'),
                        capprops=dict(color="blue"), zorder=3, showfliers=False, ax=ax)
    if medians:
        for i in range(0, data1.shape[1]):
            ax.plot(np.linspace(-0.4, 0.4, 3) + i, np.nanmedian(data1_surr.iloc[:, i]) * np.ones(3), ls=':',
                       color='gray', lw=5)

    for i in range(data1.shape[1]):
        if b_fin.iloc[:, i].values == 'c_Choice':
            mybox = cajas.artists[i]
            mybox.set_facecolor(colors[1])
            mybox.set_edgecolor(colors[1])
            mybox.set_alpha(0.5)
            for j in range(i * 5, i * 5 + 5):
                line = cajas.lines[j]
                line.set_color(colors2[1])
                line.set_mfc(colors2[1])
                line.set_mec(colors2[1])
        elif b_fin.iloc[:, i].values == 'b_Both':
            mybox = cajas.artists[i]
            mybox.set_facecolor(colors[2])
            mybox.set_edgecolor(colors[2])
            mybox.set_alpha(0.5)
            for j in range(i * 5, i * 5 + 5):
                line = cajas.lines[j]
                line.set_color(colors2[2])
                line.set_mfc(colors2[2])
                line.set_mec(colors2[2])
        elif b_fin.iloc[:, i].values == 'u_Unrelated':
            mybox = cajas.artists[i]
            mybox.set_facecolor(colors[3])
            mybox.set_edgecolor(colors[3])
            mybox.set_alpha(0.5)
            for j in range(i * 5, i * 5 + 5):
                line = cajas.lines[j]
                line.set_color(colors2[3])
                line.set_mfc(colors2[3])
                line.set_mec(colors2[3])
    plt.xticks(ticks=np.arange(0, data1.shape[1]), labels=data1.columns, rotation=30, fontsize=26)
    plt.subplots_adjust(top=0.98,bottom=0.035,left=0.01,right=0.99,hspace=0.2,wspace=0.7)

def plotSubsam(axes, v_list, spec_idx, b_fin, colors, colors2,numSamp):
    d1med = np.zeros((numSamp,len(b_fin.columns),3))
    d2med = np.zeros((numSamp,len(b_fin.columns),3))
    counter = 0
    for k in np.arange(0,6,2):
        for bb in range(len(b_fin.columns)):
            ax = axes.flat[counter]
            data1 = pd.DataFrame(v_list[k][bb]).T
            data2 = pd.DataFrame(v_list[k+1][bb]).T
            for i in range(data1.shape[1]):
                d1 = data1.iloc[:,i]
                d2 = data2.iloc[:,i]
                d1med[i,bb,int(k/2)] = np.nanmedian(d1)
                d2med[i,bb,int(k/2)] = np.nanmedian(d2)
            if b_fin.iloc[:, bb].values == 'a_Visual':
                color = colors[0]
                color2 = colors2[0]
            elif b_fin.iloc[:, bb].values == 'c_Choice':
                color = colors[1]
                color2 = colors2[1]
            elif b_fin.iloc[:, bb].values == 'b_Both':
                color = colors[2]
                color2 = colors2[2]
            elif b_fin.iloc[:, bb].values == 'u_Unrelated':
                color = colors[3]
                color2 = colors2[3]
            ax.plot(np.arange(0,numSamp,1),d1med[:,bb,int(k/2)], color = color, lw = 2)
            ax.plot(np.arange(0,numSamp,1),d2med[:,bb,int(k/2)], color = color2, lw = 2)
            ax.set_xticks(ticks = np.arange(0,10,9))
            if bb == 0:
                if k != 4:
                    ax.set_ylabel('Correlation')
                else:
                    ax.set_ylabel('SpecIdx')
            else:
                ax.set_ylabel('')
            if k != 4:
                ax.set_ylim([0.2, 1])
            else:
                ax.set_ylim([0, 0.07])
            if k!= 4:
                ax.set_xticklabels('')
            else:
                ax.set_xticklabels(labels=['N/10', 'N'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            counter += 1

    for bb in range(len(b_fin.columns)):
        ax = axes.flat[counter]
        data1 = pd.DataFrame(spec_idx[bb]).T
        d1med2 = np.zeros(data1.shape[1])
        for i in range(data1.shape[1]):
            d1 = data1.iloc[:,i]
            d1med2[i] = np.nanmedian(d1)
        if b_fin.iloc[:, bb].values == 'a_Visual':
            color = colors[0]
        elif b_fin.iloc[:, bb].values == 'c_Choice':
            color = colors[1]
        elif b_fin.iloc[:, bb].values == 'b_Both':
            color = colors[2]
        elif b_fin.iloc[:, bb].values == 'u_Unrelated':
            color = colors[3]
        ax.plot(np.arange(0,numSamp,1),d1med2, color=color, lw=2)
        ax.set_xticks(ticks=np.arange(0, numSamp, numSamp-1))
        ax.set_ylabel('SpecIdx')
        ax.set_xticklabels('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0, 0.07])
        counter += 1
    plt.subplots_adjust(top=0.98,bottom=0.035,left=0.01,right=0.99,hspace=0.2,wspace=0.7)

def plotEffSizes(ax, data1, data2, b_fin, colors, colors2):
    size = np.zeros(data1.shape[1])
    for i in range(data1.shape[1]):
        d1 = data1.iloc[:, i]
        d1 = d1[~np.isnan(d1)]
        d2 = data2.iloc[:, i]
        d2 = d2[~np.isnan(d2)]
        if np.nanmedian(d2) > np.nanmedian(d1):
            size[i] = sts.mannwhitneyu(d1, d2).statistic / (d1.shape[0] * d2.shape[0])
        elif np.nanmedian(d2) < np.nanmedian(d1):
            size[i] = 1 - (sts.mannwhitneyu(d1, d2).statistic / (d1.shape[0] * d2.shape[0]))
    for i in range(data1.shape[1]):
        if b_fin.iloc[:, i].values == 'a_Visual':
            cc = colors[0]
            cc2 = colors2[0]
        elif b_fin.iloc[:, i].values == 'c_Choice':
            cc = colors[1]
            cc2 = colors2[1]
        elif b_fin.iloc[:, i].values == 'b_Both':
            cc = colors[2]
            cc2 = colors2[2]
        elif b_fin.iloc[:, i].values == 'u_Unrelated':
            cc = colors[3]
            cc2 = colors2[3]
        ax.scatter(x=i, y=size[i], alpha=0.5, s=100, color=cc, edgecolor=cc2)
    ax.set_ylim([0, 1])
    ax.axhline(0.5, lw=2, color='gray', alpha=0.5, ls='--')
    ax.set_ylabel(r"Cliff's $\delta$")
    plt.xticks(ticks=np.arange(0, data1.shape[1]), labels=data1.columns, rotation=30, fontsize=26)

def plotScatterRemoval(ax, data1, data2, b_fin, colors, colors2, ylab):
    counter = 0
    for i in range(data1.shape[1]):
        if b_fin.iloc[:, i].values == 'a_Visual':
            color = colors[0]
        elif b_fin.iloc[:, i].values == 'c_Choice':
            color = colors[1]
        elif b_fin.iloc[:, i].values[0] == 'b_Both':
            color = colors[2]
        elif b_fin.iloc[:, i].values[0] == 'u_Unrelated':
            color = colors[3]
        plt.scatter(counter + np.random.uniform(-0.15, 0.15, data1.iloc[:, i].shape[0]), data1.iloc[:, i],
                    color=color, alpha=0.3)
        counter += 1
    counter = 0
    for i in range(data1.shape[1]):
        if b_fin.iloc[:, i].values == 'a_Visual':
            color = colors2[0]
        elif b_fin.iloc[:, i].values == 'c_Choice':
            color = colors2[1]
        elif b_fin.iloc[:, i].values[0] == 'b_Both':
            color = colors2[2]
        elif b_fin.iloc[:, i].values[0] == 'u_Unrelated':
            color = colors2[3]
        plt.scatter(0.4 + counter + np.random.uniform(-0.15, 0.15, data2.iloc[:, i].shape[0]), data2.iloc[:, i],
                    color=color, alpha=0.3)
        counter += 1

    if ylab == 'Correlation':
        ax.set_ylabel(r"$(cor_{pop} - cor_{remov})/N$", fontsize=22)
        ax.set_ylim([-0.1, 0.11])
    elif ylab == 'Spec Idx':
        ax.set_ylabel(r"$(SpecIdx_{pop} - SpecIdx_{remov})/N$", fontsize=22)
        ax.set_ylim([-0.04, 0.04])
    plt.xticks(ticks=np.arange(0, data1.shape[1]), labels=data1.columns, rotation=30, fontsize=26)
