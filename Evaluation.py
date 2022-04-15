import gzip
import pickle
import os
import networkx as nx

import numpy as np
import seaborn

from utils import *
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import sem
from wordcloud import WordCloud
import multidict as multidict
import re
import datetime
import matplotlib

from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad


def ensureFolder(folder):
    curfol = "./"
    for fol in folder.split("/")[:-1]:
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

def normAxis(mat, axes):
    idx = []
    axes = tuple(axes)
    for i in range(len(mat.shape)):
        if i in axes or (i==len(mat.shape)-1 and -1 in axes):
            idx.append(None)
        else:
            idx.append(slice(mat.shape[i]))
    idx = tuple(idx)
    newMat = (mat-np.min(mat, axis=axes)[idx])/(np.max(mat, axis=axes)[idx]-np.min(mat, axis=axes)[idx]+1e-20)
    return newMat

def nansem(x):
    x = np.array(x)
    return sem(x[~np.isnan(x)])

def readObservations(folder, name_ds, output_folder, onlyreturnindexwds=False):
    dataFile = folder+name_ds
    observations = []
    wdToIndex, index = {}, 0


    indexToWd = {}
    with open(output_folder+name_ds.replace("_events.txt", "")+"_indexWords.txt", "r", encoding="utf-8") as f:
        for line in f:
            index, wd = line.replace("\n", "").split("\t")
            indexToWd[int(index)] = wd
    if onlyreturnindexwds: return indexToWd

    index = 0
    with open(dataFile, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.replace("\n", "").split("\t")
            timestamp = float(l[0])
            words = l[1].split(",")
            try:
                clusTmp = int(float(l[2]))  # See generate data/save() -> temporal cluster saved before textual
                clusTxt = int(float(l[3]))
            except:
                clusTxt = None
                clusTmp = None
            uniquewords, cntwords = np.unique(words, return_counts=True)
            for un in uniquewords:
                if un not in wdToIndex:
                    wdToIndex[un] = index
                    index += 1
            uniquewords = [wdToIndex[un] for un in uniquewords]
            uniquewords, cntwords = np.array(uniquewords, dtype=int), np.array(cntwords, dtype=int)

            tup = (i, timestamp, [uniquewords, cntwords], [clusTxt, clusTmp])
            observations.append(tup)

            if 'Covid' in dataFile and i>10000 and False:  # ============================================
                print("BROKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEN")
                break

            if i > 5000 and False:
                print("BROKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEN")  # ============================================
                break

    V = len(indexToWd)
    observations = np.array(observations, dtype=object)

    return observations, V, indexToWd

def getData(params):
    (folder, DS, nbClasses, run_time, multivariate,
     overlap_voc, overlap_temp, perc_rand,
     voc_per_class, words_per_obs, theta0,
     lamb0_poisson, lamb0_classes, alpha0, means, sigs) = params

    name_ds = f"Obs_nbclasses={nbClasses}_lg={run_time}_overlapvoc={overlap_voc}_overlaptemp={overlap_temp}" \
              f"_percrandomizedclus={perc_rand}_vocperclass={voc_per_class}_wordsperevent={words_per_obs}_DS={DS}"+"_events.txt"
    output_folder = folder.replace("data/", "output/")

    observations, vocabulary_size, indexToWd = readObservations(folder, name_ds, output_folder)

    return name_ds, observations, vocabulary_size

def read_particles(folderOut, nameOut, time=-1):
    if time==-1: txtTime = "_final"
    else: txtTime = f"_obs={int(time)}"

    with gzip.open(folderOut+nameOut+txtTime+"_particles.pkl.gz", "rb") as f:
        DHP = pickle.load(f)

    # popClus, cnt = np.unique(DHP.particles[0].docs2cluster_ID, return_counts=True)
    # selected_clus = [clus for _, clus in sorted(zip(cnt, popClus), reverse=True)][:20]
    # print(sorted(cnt, reverse=True)[:50])

    return DHP

def fill_clusters(DHP, consClus):
    dicFilesClus = {}
    for i in range(len(DHP.particles)):
        for clus_num, (clus_index, file_cluster) in enumerate(DHP.particles[i].files_clusters):
            if file_cluster not in dicFilesClus:
                if clus_index in consClus:
                    dicFilesClus[file_cluster] = read_clusters(file_cluster)

    for i in range(len(DHP.particles)):
        for clus_num, (clus_index, file_cluster) in enumerate(DHP.particles[i].files_clusters):
            if clus_index in consClus:
                clus = dicFilesClus[file_cluster]
                DHP.particles[i].clusters[clus_index] = clus

    return DHP

def read_clusters(file_cluster):
    with gzip.open(file_cluster, "rb") as f:
        cluster = pickle.load(f)

    return cluster

# =========== PLOTS =============

# Wordsclouds
def getFrequencyDictForText(words):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in words:
        if re.match("a|the|an|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict

def makeWordCloud(dictWordsFreq, onlyreturnImg=False, colormap="twilight", nbWds = 200):
    #alice_mask = np.array(Image.open("alice_mask.png"))

    x, y = np.ogrid[:1000, :1000]
    mask = (x - 500) ** 2 + (y - 500) ** 2 > 450 ** 2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color='white', max_words=nbWds, mask=mask, contour_width=10, contour_color='black', colormap=colormap)  #"cividis")
    # generate word cloud
    wc.generate_from_frequencies(dictWordsFreq)

    if onlyreturnImg:
        return wc

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.tight_layout()
    plt.axis("off")

def drawCirc(ax,radius,centX,centY,angle_,theta2_,width, widtharrow, alpha=1.,color_='black',zorder=0):
    #========Line
    arc = Arc((centX,centY),radius,radius,angle=angle_,lw=width,
              theta1=0,theta2=theta2_,capstyle='round',linestyle='-',color=color_, alpha=alpha,zorder=zorder)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            widtharrow,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_,
            alpha=alpha,
            zorder=zorder
        )
    )
    #ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius])
    # Make sure you keep the axes scaled or else arrow will distort

# Precompute matrices
def plotAdjTrans(results_folder, name_output, DHP, indexToWd, observations, consClus):
    clusToInd = {int(c): i for i,c in enumerate(consClus)}
    with open(results_folder+name_output+"_clusToInd.pkl", 'wb') as f:
        pickle.dump(clusToInd, f)

    A = np.zeros((len(consClus), len(consClus), len(means)))
    transparency = np.zeros((len(consClus), len(consClus), len(means)))
    transparency_permonth = np.zeros((12, len(consClus), len(consClus), len(means)))
    for indexclus, c in enumerate(consClus):
        #print("Compute adjacency/weigths matrix", indexclus, "/", len(consClus))


        active_timestamps = np.array(list(zip(DHP.particles[0].docs2cluster_ID, observations[:, 1])))
        active_timestamps = np.array([ats for ats in active_timestamps if ats[0] in consClus])
        weigths = np.zeros((len(consClus), len(means)))
        div = 0
        weigths_permonth = np.zeros((12, len(consClus), len(means)))
        div_permonth = np.zeros((12))

        for i, t in enumerate(active_timestamps[active_timestamps[:, 0]==c][:, 1]):
            active_timestamps_cons = active_timestamps[active_timestamps[:, 1]>active_timestamps[i, 1]-np.max(means)-np.max(sigs)]
            active_timestamps_cons = active_timestamps_cons[active_timestamps_cons[:, 1]<active_timestamps[i,1]]
            timeseq = active_timestamps_cons[:, 1]
            clusseq = active_timestamps_cons[:, 0]
            month = datetime.datetime.fromtimestamp(t*60).month-1  # Bc month=index, begins at 0
            div += 1  # Intensité subie par observation
            div_permonth[month] += 1
            if len(timeseq)<=0: continue
            time_intervals = timeseq[-1] - timeseq[:-1]
            if len(time_intervals)!=0:
                RBF = RBF_kernel(means, time_intervals, sigs)  # num_time_points, size_kernel
                for (clus, trig) in zip(clusseq, RBF):
                    clus = int(clus)
                    if clus not in DHP.particles[0].clusters[c].alpha_final:
                        continue
                    indclus = clusToInd[clus]

                    weigths[indclus] += trig*DHP.particles[0].clusters[c].alpha_final[clus]
                    weigths_permonth[month, indclus] += trig*DHP.particles[0].clusters[c].alpha_final[clus]


        weigths /= (div+1e-20)  # Only Siths deal in absolutes (well maybe not)
        weigths_permonth /= (div_permonth[:, None, None]+1e-20)  # Only Siths deal in absolutes (again)

        for c2 in DHP.particles[0].clusters[c].alpha_final:
            if c2 not in consClus: continue

            for l in range(len(means)):
                A[clusToInd[c],clusToInd[c2], l] = DHP.particles[0].clusters[c].alpha_final[c2][l]
                transparency[clusToInd[c],clusToInd[c2], l] = weigths[clusToInd[int(c2)], l]

                for month in range(len(transparency_permonth)):
                    transparency_permonth[month, clusToInd[c],clusToInd[c2], l] = weigths_permonth[month, clusToInd[int(c2)], l]

    np.save(results_folder+name_output+"_adjacency", A)
    np.save(results_folder+name_output+"_transparency", transparency)
    np.save(results_folder+name_output+"_transparency_permonth", transparency_permonth)



# Kernel
def plot_kernel(c, A, transparency, DHP, observations, consClus, clusToInd):
    A, transparency = A.copy(), transparency.copy()

    transparency = transparency.sum(axis=-1)
    transparency = normAxis(transparency, axes=[1])
    transparency = transparency**2

    dt = np.linspace(0, np.max(means)+np.max(sigs), 1000)
    trigger = RBF_kernel(means, dt, sigs)
    for index, influencer_clus in enumerate(consClus):
        if influencer_clus not in DHP.particles[0].clusters[c].alpha_final:
            continue
        trigger_clus = trigger.dot(A[clusToInd[c],clusToInd[influencer_clus]])

        transp = np.sum(transparency[clusToInd[c], clusToInd[influencer_clus]])
        if transp<0.2: continue

        plt.plot(dt, trigger_clus, "-", label=f"Cluster {influencer_clus}", alpha=transp, c=f"C{index}")
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)

# Real timeline
def plot_real_timeline(c, DHP, observations, consClus, transparency, clusToInd, datebeg="01/08/21"):

    transparency = transparency.copy()

    transparency = transparency.sum(axis=-1)
    transparency = normAxis(transparency, axes=[1])
    transparency = transparency**3

    active_timestamps = np.array(list(zip(DHP.particles[0].docs2cluster_ID, observations[:, 1])))
    active_timestamps = np.array([ats for ats in active_timestamps if ats[0] in consClus])
    indToClus = {int(c): i for i,c in enumerate(consClus)}
    times_cons = []
    dt = 15  # 1 obs every 15 minute
    if len(active_timestamps)==0:
        return
    for ind, x_i in enumerate(active_timestamps[active_timestamps[:, 0]==c][:, 1]):
        if len(times_cons) == 0:
            times_cons.append(x_i)
        elif x_i>times_cons[-1]+dt:
            times_cons.append(x_i)

    #times_cons = np.linspace(np.min(active_timestamps[:, 1]), np.max(active_timestamps[:, 1]), 1000)
    #plt.plot([np.min(active_timestamps[active_timestamps[:, 0]==c][:, 1])]*2, [0,1], "-k")

    array_intensity = []
    array_times = []
    for t in times_cons:
        active_timestamps_cons = active_timestamps[active_timestamps[:, 1]>t-np.max(means)-np.max(sigs)]
        active_timestamps_cons = active_timestamps_cons[active_timestamps_cons[:, 1]<t]
        timeseq = active_timestamps_cons[:, 1]
        clusseq = active_timestamps_cons[:, 0]
        unweighted_triggering_kernel = np.zeros((len(consClus), len(means)))
        if len(timeseq)<=1: continue
        time_intervals = timeseq[-1] - timeseq[timeseq<timeseq[-1]]

        RBF = RBF_kernel(means, time_intervals, sigs)  # num_time_points, size_kernel
        for (clus, trig) in zip(clusseq, RBF):
            indclus = indToClus[int(clus)]
            unweighted_triggering_kernel[indclus] += trig

        alpha_recomp = np.zeros(shape=unweighted_triggering_kernel.shape)
        for clus in set(clusseq):
            if clus not in DHP.particles[0].clusters[c].alpha_final:
                continue
            indclus = indToClus[int(clus)]
            alpha_recomp[indclus] = DHP.particles[0].clusters[c].alpha_final[clus]

        array_times.append(t)
        array_intensity.append(np.sum(unweighted_triggering_kernel*alpha_recomp, axis=1))


    array_intensity = np.array(array_intensity)
    plotted = False
    for influencer_clus in consClus:
        if len(array_intensity)>0:
            plt.plot(array_times, array_intensity[:, indToClus[influencer_clus]], "-", label=f"Cluster {influencer_clus}", alpha=transparency[indToClus[c], indToClus[influencer_clus]])
            plotted = True
    #plt.legend()
    if not plotted:
        return

    limleft = datetime.datetime.timestamp(datetime.datetime.strptime(datebeg, '%d/%m/%y'))/60  # In minutes
    limright = datetime.datetime.timestamp(datetime.datetime.fromtimestamp(np.max(active_timestamps[:, 1])))
    plt.xlim(left=limleft, right=limright)

    dt = 30.5*24*60  # day
    x_ticks = [limleft]
    while x_ticks[-1]<np.max(array_times):
        x_ticks.append(x_ticks[-1]+dt)
    x_labels = [datetime.datetime.fromtimestamp(float(ts)*60).strftime("%d %b") for ts in x_ticks]
    plt.xticks(x_ticks, x_labels, rotation=45, ha="right")

# Plot each cluster
def plotIndividualClusters(A, transparency, results_folder, namethres, DHP, indexToWd, observations, consClus, clusToInd):
    scale = 4
    for c in sorted(DHP.particles[0].clusters, reverse=False):
        if len(DHP.particles[0].clusters[c].alpha_final)==0:  # Not present for r=0
            continue
        if c not in consClus:
            continue


        fig = plt.figure(figsize=(1*scale, 3*scale))

        #print(f"Cluster {c}a")
        plt.subplot(3,1,1)
        plt.title(f"Cluster {c} - {DHP.particles[0].docs2cluster_ID.count(c)} obs")
        wds = {indexToWd[idx]: count for count, idx in reversed(sorted(zip(DHP.particles[0].clusters[c].word_distribution, list(range(len(DHP.particles[0].clusters[c].word_distribution)))))) if count!=0}
        makeWordCloud(wds)

        #print(f"Cluster {c}b")
        plt.subplot(3,1,2)
        plot_kernel(c, A, transparency, DHP, observations, consClus, clusToInd)

        #print(f"Cluster {c}c")
        plt.subplot(3, 1, 3)
        plot_real_timeline(c, DHP, observations, consClus, transparency, clusToInd, datebeg="01/01/19")

        fig.tight_layout()
        plt.savefig(results_folder+f"/Clusters/{namethres}_cluster_{c}.pdf")
        #plt.show()
        plt.close()


# Plot... Graphe topics, bah oui
def plotGraphGlob(A, transparency, results_folder, name_output, DHP, indexToWd, consClus, clusToInd, ax_ext=None, ax_clus=None, colorClus = None, axesNorm=None):
    if colorClus is None:
        colorClus = {int(c): f"C{i}" for i,c in enumerate(consClus)}

    A, transparency = A.copy(), transparency.copy()

    if axesNorm is None:
        axesNorm = [2]
    transparency = normAxis(transparency, axes=axesNorm)
    A = normAxis(A, axes=axesNorm)

    cmap = matplotlib.cm.get_cmap('afmhot')
    scale = len(means)*5
    scaleedge = 0.01

    if ax_ext is None:
        fig, ax = plt.subplots(1,len(means), figsize=(len(means)*scale, 1*scale), subplot_kw=dict(box_aspect=1))
        num = len(consClus)
        nbx = int(num**0.5)+1
        nby = nbx
        fig_clus, ax_clus = plt.subplots(nbx, nby, figsize=(scale*nbx, scale*nby))
        ax_clus = ax_clus.flat
    else:
        ax = ax_ext

    G = nx.from_numpy_array(A.sum(axis=-1))
    pos = nx.spring_layout(G, k=4)
    for l in range(len(means)):
        radius = 0.2
        minx = np.min([pos[index][0] for index in pos])-radius
        maxx = np.max([pos[index][0] for index in pos])+radius
        miny = np.min([pos[index][1] for index in pos])-radius
        maxy = np.max([pos[index][1] for index in pos])+radius
        for c in consClus:
            #print("Kernel", l, "- C -", clusToInd[c], "/", len(consClus))

            xnode, ynode = pos[clusToInd[c]]
            ax[l].plot([xnode], [ynode], "o", markersize = radius*1000, c=colorClus[c])

            if not bool(ax_clus[clusToInd[c]].get_images()):
                wds = {indexToWd[idx]: count for count, idx in reversed(sorted(zip(DHP.particles[0].clusters[c].word_distribution, list(range(len(DHP.particles[0].clusters[c].word_distribution)))))) if count!=0}
                imgWC = makeWordCloud(wds, onlyreturnImg=True, colormap="cividis")
                imgWC=imgWC.to_array()

                alpha = ~np.all(imgWC == 255, axis=2) * 255
                rgbaimgWC = np.dstack((imgWC, alpha)).astype(np.uint8)

                bbox = [radius, 2*radius, -radius/2, radius/2]
                ax_clus[clusToInd[c]].imshow(rgbaimgWC, interpolation="nearest", extent=bbox, zorder=1)
                ax_clus[clusToInd[c]].plot(radius/2, 0, "o", markersize=radius*1500, color=colorClus[c])
                ax_clus[clusToInd[c]].text(radius*8/10, 0, "=", fontsize=200, ha="center", va="center")
                ax_clus[clusToInd[c]].set_xlim([0, 2*radius])
                ax_clus[clusToInd[c]].set_ylim([-radius/2, radius/2])
                ax_clus[clusToInd[c]].set_xticks([])
                ax_clus[clusToInd[c]].set_yticks([])

            for c2 in consClus:
                transp = transparency[clusToInd[c], clusToInd[c2], l]
                color = cmap(A[clusToInd[c], clusToInd[c2], l])

                if c==c2:
                    costheta = np.cos(45)
                    sintheta = np.sin(45)
                    if c>c2:
                        shift=0.01
                    else:
                        shift=-0.01
                    shiftx = shift*np.cos(np.arccos(costheta)+np.pi/2)
                    shifty = shift*np.sin(np.arcsin(sintheta)+np.pi/2)
                    x = xnode+radius*costheta+shiftx
                    y = ynode+radius*sintheta+shifty

                    w = 1000*scaleedge
                    drawCirc(ax[l],0.15,x,y,-60,215,width=w,widtharrow=1.5*w/500, color_=color, alpha=transp, zorder=10)

                if c!=c2:
                    xnode2, ynode2 = pos[clusToInd[c2]]
                    dx = xnode2-xnode
                    dy = ynode2-ynode
                    theta = np.arctan2(dy, dx)
                    costheta = np.cos(theta)
                    sintheta = np.sin(theta)

                    if theta>np.pi:
                        coeff=1
                    else:
                        coeff=-1
                    shift=0.05

                    shiftx = shift*np.cos(np.pi/2-coeff*theta)
                    shifty = shift*np.sin(np.pi/2-coeff*theta)

                    x = xnode+radius*costheta+shiftx
                    dx -= 2*radius*costheta
                    y = ynode+radius*sintheta+shifty
                    dy -= 2*radius*sintheta

                    ax[l].arrow(x, y, dx, dy, width=1.*scaleedge, zorder=10, length_includes_head=True, color=color, alpha=transp)


        ax[l].set_xlim([minx, maxx])
        ax[l].set_ylim([miny, maxy])
        ax[l].set_xticks([])
        ax[l].set_yticks([])

    #plt.tight_layout()
    if ax_ext is None:
        plt.figure(fig.number)
        plt.savefig(results_folder+name_output+"_Recap.pdf")
        plt.close(fig)
        plt.figure(fig_clus.number)
        plt.savefig(results_folder+name_output+"_Recap_clus.jpg")
        plt.close(fig_clus)

def plotGraphGlobEveryMonth(observations, A, transparency, transparency_permonth, results_folder, name_output, DHP, indexToWd, consClus, clusToInd, axesNorm=None, numClusPerMonth=5):
    popClus_month = {}
    for c, o in zip(DHP.particles[0].docs2cluster_ID, observations):
        if c not in consClus: continue
        t = o[1]
        month = datetime.datetime.fromtimestamp(t*60).month
        if month not in popClus_month: popClus_month[month] = {}
        if c not in popClus_month[month]: popClus_month[month][c] = 0

        popClus_month[month][c] += 1

    scale = len(means)*5
    fig, ax = plt.subplots(len(popClus_month), len(means), figsize=(len(means)*scale, len(popClus_month)*scale), subplot_kw=dict(box_aspect=1))

    allConsClus = set()
    for month in sorted(popClus_month):
        entries_to_cons = list(sorted(popClus_month[month].items(), key=lambda x: x[1], reverse=True)[:numClusPerMonth])  # Select top 5 clusters
        for clus, cnt in entries_to_cons:
            allConsClus.add(clus)

    num = len(allConsClus)
    nbx = int(num**0.5)+1
    nby = nbx
    fig_clus, ax_clus = plt.subplots(nbx, nby, figsize=(scale*nbx, scale*nby))
    ax_clus = ax_clus.flat

    if len(popClus_month)==1: ax=np.array([ax])
    indexes_month = list(sorted(popClus_month))
    for i_month, month in enumerate(sorted(popClus_month)):
        entries_to_cons = list(sorted(popClus_month[month].items(), key=lambda x: x[1], reverse=True)[:numClusPerMonth])  # Select top 5 clusters
        consClus_month = []
        for clus, cnt in entries_to_cons:
            consClus_month.append(clus)

        consClus_month_index = [clusToInd[clusmonth] for clusmonth in consClus_month]
        clusToInd_month = {clusmonth: i for i, clusmonth in enumerate(consClus_month)}
        print(A.shape, transparency.shape, transparency_permonth.shape, np.shape(ax), month-1, consClus_month_index)
        plotGraphGlob(A[consClus_month_index][:, consClus_month_index], transparency_permonth[month-1][consClus_month_index][:, consClus_month_index], results_folder, name_output, DHP, indexToWd, consClus_month, clusToInd_month, ax_ext=ax[i_month], ax_clus=ax_clus, axesNorm=axesNorm)

    monthIndex = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i,j].set_ylabel(monthIndex[indexes_month[i]-1], fontsize=4*scale)
            ax[i,j].set_xlabel(f"Influence at {means[j]}min", fontsize=4*scale)

    for a in ax.flat:
        a.label_outer()

    plt.figure(fig.number)
    plt.savefig(results_folder+name_output+"_Recap_months.pdf")
    plt.close(fig)
    plt.figure(fig_clus.number)
    fig_clus.savefig(results_folder+name_output+"_Recap_months_clus.jpg")
    plt.close(fig_clus)


# Plot... Timeline!
def plotTimeline(observations, results_folder, name_output, DHP, indexToWd, consClus, numClusPerMonth=5):
    res = 12*30/3
    res = int(res)

    popClus_month = {}
    for c, o in zip(DHP.particles[0].docs2cluster_ID, observations):
        if c not in consClus: continue
        t = o[1]
        month = datetime.datetime.fromtimestamp(t*60).month
        if month not in popClus_month: popClus_month[month] = {}
        if c not in popClus_month[month]: popClus_month[month][c] = 0

        popClus_month[month][c] += 1

    allConsClus = set()
    for month in sorted(popClus_month):
        entries_to_cons = list(sorted(popClus_month[month].items(), key=lambda x: x[1], reverse=True)[:numClusPerMonth])  # Select top 5 clusters
        for clus, cnt in entries_to_cons:
            allConsClus.add(clus)
    allConsClus = list(allConsClus)
    allConsClus = list(sorted(allConsClus, key=lambda x: DHP.particles[0].docs2cluster_ID.count(x),reverse=True))

    clusToInd_heatmap = {clus: i for i, clus in enumerate(allConsClus)}

    matheatmap = np.zeros((len(allConsClus), res))

    tsclus = {}
    for c, o in zip(DHP.particles[0].docs2cluster_ID, observations):
        if c not in allConsClus: continue
        if c not in tsclus: tsclus[c]=[]
        tsclus[c].append(o[1])

    minTime, maxTime = 1e20, -1e20
    for c in tsclus:
        if len(tsclus[c])==0:
            tsclus[c] = np.array([])
            continue
        if np.min(tsclus[c])<minTime: minTime=np.min(tsclus[c])
        if np.max(tsclus[c])>maxTime: maxTime=np.max(tsclus[c])
    if minTime>maxTime:
        minTime=-1
        maxTime=1
    bins = np.linspace(minTime, maxTime, res+1)
    for c in sorted(tsclus):
        n, _, _ = plt.hist(tsclus[c], bins=bins, density=False)
        plt.close()
        matheatmap[clusToInd_heatmap[c]] = n

    topWords = ["" for _ in range(len(clusToInd_heatmap))]
    for c in sorted(tsclus):
        wds = [indexToWd[idx] for count, idx in reversed(sorted(zip(DHP.particles[0].clusters[c].word_distribution, list(range(len(DHP.particles[0].clusters[c].word_distribution)))))) if count!=0]
        topWords[clusToInd_heatmap[c]] = ", ".join(wds[:5])

    scale=2

    for normornot in ["norm", "abs"]:
        fig = plt.figure(figsize=(15*scale, 0.2*scale*len(allConsClus)))

        if normornot=="norm":
            matheatmap_used = matheatmap/np.max(matheatmap, axis=1)[:, None]
        else:
            matheatmap_used = matheatmap
        seaborn.heatmap(matheatmap_used, cmap="afmhot_r", square=False, cbar=False)


        xticks = [i for i in range(len(bins)) if i%(res//10)==0]
        xlabels = [datetime.datetime.fromtimestamp(bins[i]*60).strftime("%d %b") for i in xticks]

        plt.yticks(np.array(list(range(len(tsclus))))+0.5, topWords, rotation=0, fontsize=11*scale)
        plt.xticks(xticks, xlabels, rotation=45, fontsize=11*scale)

        # radius = 0.5
        # for i, c in enumerate(allConsClus):
        #     print(i,c)
        #     wds = {indexToWd[idx]: count for count, idx in reversed(sorted(zip(DHP.particles[0].clusters[c].word_distribution, list(range(len(DHP.particles[0].clusters[c].word_distribution)))))) if count!=0}
        #     wc = makeWordCloud(wds, onlyreturnImg=True, nbWds = 20)
        #     wc = wc.to_array()
        #     alpha = ~np.all(wc == 255, axis=2) * 255
        #     rgbaimgWC = np.dstack((wc, alpha)).astype(np.uint8)
        #     bbox = [-2*radius, 0, i-0.5-radius, i-0.5+radius]
        #     plt.imshow(rgbaimgWC, interpolation="nearest", extent=bbox, zorder=1)


        plt.tight_layout()
        plt.savefig(results_folder+name_output+f"_Timeline_{normornot}.pdf")
        #plt.show()
        plt.close()


def metrics(A, transparency, DHP, clusToInd):
    effective_interaction = transparency.copy()
    for c in clusToInd:
        effective_interaction[clusToInd[c]] -= lamb0_poisson/DHP.particles[0].docs2cluster_ID.count(c)
    mean = np.sum((A-lamb0_poisson)*transparency, axis=(0,1))/np.sum((A-lamb0_poisson), axis=(0,1))
    print(mean)
    mean = np.sum((A-lamb0_poisson), axis=(0,1))/np.sum(transparency**0, axis=(0,1))
    print(mean)
    mean = np.sum(transparency, axis=(0,1))/np.sum(transparency**0, axis=(0,1))
    print(mean)

if __name__=="__main__":
    try:
        RW = sys.argv[1]
        XP = sys.argv[2]
    except:
        RW = "1"
        XP = "osef"



    if RW=="0":
        if True:
            num_NMI_last = 5000000000  # Consider all observations
            norm_err = 0.2  # For visual representation or error bars, not used


            nbClasses = 2
            num_obs = 5000

            overlap_voc = 0.  # Proportion of voc in common between a clusters and its direct neighbours
            overlap_temp = 0.  # Overlap between the kernels of the simulating process
            perc_rand = 0.  # Percentage of events to which assign random textual cluster

            voc_per_class = 1000  # Number of words available for each cluster
            words_per_obs = 20  # Twitter or headlines typically have few named entities

            lamb0_poisson = 0.01  # Cannot be inferred
            lamb0_classes = 0.1  # Cannot be inferred
            theta0 = 10.  # Has already been documented for RW in LDA like models, DHP, etc ~0.01, 0.001 ; here it's 10 to ease computing the overlap_voc
            alpha0 = 1.  # Uniform beta or Dirichlet prior
            means = np.array([3, 7, 11])
            sigs = np.array([1., 1., 1.])

            arrR = [0., 1.]
            nbDS = 10
            sample_num = 2000  # Typically 5 active clusters, so 25*5 parameters to infer using 2000*5 samples => ~80 samples per parameter
            particle_num = 10  # Like 10 simultaneous runs
            multivariate = True
            printRes = True
            eval_on_go = True

            folder = "data/Synth/"
            output_folder = "output/Synth/"

        # Overlap voc vs overlap temp
        def XP1(folder, output_folder):
            folder += "XP1/"
            output_folder_based = output_folder + "XP1/"

            for model in ["MPDHP", "PDHP", "PDP", ]:
                output_folder = output_folder_based + model + "/"
                results_folder = output_folder.replace("output/", "results/")
                ensureFolder(results_folder)

                overlaps_voc = np.linspace(0, 1, 6)
                overlaps_temp = np.linspace(0, 1, 6)

                matRes = np.empty((len(arrR), len(overlaps_voc), len(overlaps_temp)))
                matRes[:] = np.nan
                matStd = np.empty((len(arrR), len(overlaps_voc), len(overlaps_temp)))
                matStd[:] = np.nan
                lab_overlap_voc = {}
                lab_overlap_temp = {}


                r_ref = 1.
                index_req1 = None
                for i_r, r in enumerate(arrR):
                    if r<r_ref+0.01 and r>r_ref-0.01: index_req1 = i_r

                for i_overlap_voc, overlap_voc in enumerate(sorted(overlaps_voc)):
                    for i_overlap_temp, overlap_temp in enumerate(sorted(overlaps_temp)):
                        overlap_voc = np.round(overlap_voc, 2)
                        overlap_temp = np.round(overlap_temp, 2)

                        lab_overlap_voc[i_overlap_voc] = str(overlap_voc)
                        lab_overlap_temp[i_overlap_temp] = str(overlap_temp)

                        arrResR = [[] for _ in range(len(arrR))]
                        for DS in range(nbDS):
                            params = (folder, DS, nbClasses, num_obs, multivariate,
                                      overlap_voc, overlap_temp, perc_rand,
                                      voc_per_class, words_per_obs, theta0,
                                      lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                            try:
                                name_ds, observations, vocabulary_size = getData(params)
                                name_ds = name_ds.replace("_events.txt", "")
                            except Exception as e:
                                print(f"Data not found - {e}")
                                continue

                            for i_r, r in enumerate(arrR):
                                print(f"{model} - DS {DS} - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                                r = np.round(r, 2)

                                name_output = f"{name_ds}_r={r}" \
                                              f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                              f"_samplenum={sample_num}_particlenum={particle_num}"

                                try:
                                    DHP = read_particles(output_folder, name_output, get_clusters=False)
                                except Exception as e:
                                    print(f"Output not found - {e}")
                                    arrResR[i_r].append(np.nan)
                                    continue

                                selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                                clus_inf = selected_particle.docs2cluster_ID
                                clus_true = np.array(list(map(list, observations[:, 3])))[:, 0]

                                clus_inf = clus_inf[-num_NMI_last:]
                                clus_true = clus_true[-num_NMI_last:]

                                if len(clus_inf) != len(clus_true):
                                    arrResR[i_r].append(np.nan)
                                    continue

                                score = NMI(clus_true, clus_inf)

                                arrResR[i_r].append(score)
                                print(score)

                        arrResR = np.array(arrResR)
                        for i_r, r in enumerate(arrR):
                            if not np.isnan(arrResR[i_r]).all():
                                if i_r != index_req1:
                                    arrResR[i_r] -= arrResR[index_req1]
                                meanTabNMI = np.nanmean(arrResR[i_r])
                                stdTabNMI = nansem(arrResR[i_r])
                                matRes[i_r, i_overlap_voc, i_overlap_temp] = meanTabNMI
                                matStd[i_r, i_overlap_voc, i_overlap_temp] = stdTabNMI

                        scale=6
                        plt.figure(figsize=(4*scale, 1*scale))
                        for i_r, r in enumerate(arrR):
                            plt.subplot(1, 4, i_r+1)
                            lab_x = [str(lab_overlap_voc[idx]) for idx in lab_overlap_voc]
                            lab_y = [str(lab_overlap_temp[idx]) for idx in lab_overlap_temp]

                            if i_r == index_req1:
                                sns.heatmap(np.round(matRes[i_r], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=True,
                                        cbar_kws={"label":"NMI", "shrink": 0.6}, vmin=0, vmax=1)
                                plt.title(f"(reference) r = {r}")
                            else:
                                sns.heatmap(np.round(matRes[i_r], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="RdBu_r", square=True, annot=True,
                                            cbar_kws={"label":r"$\Delta$ NMI", "shrink": 0.6}, vmin=-1, vmax=1)
                                plt.title(f"r = {r}")

                            plt.gca().invert_yaxis()
                            for ix in range(len(lab_overlap_voc)):
                                for iy in range(len(lab_overlap_temp)):
                                    col = "white"
                                    if np.round(matRes[i_r, ix, iy], 2)<0.47: col="k"
                                    plt.text(ix+0.5, iy+0.2, fr"$\pm${np.round(matStd[i_r, ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("Textual overlap")
                            plt.ylabel("Temporal overlap")
                        plt.tight_layout()
                        plt.savefig(results_folder+"heatmap.pdf")
                        plt.close()

        # nbClasses vs lamb0
        def XP2(folder, output_folder):
            folder += "XP2/"
            output_folder_based = output_folder + "XP2/"

            for model in ["MPDHP", "PDHP", "PDP", ]:
                if model != "MPDHP":
                    continue
                output_folder = output_folder_based + model + "/"
                results_folder = output_folder.replace("output/", "results/")
                ensureFolder(results_folder)

                arrNbClasses = list(range(2, 10))
                arrLambPoisson = np.logspace(-4, 1, 6)
                arrR = [1.]


                matRes = np.empty((len(arrR), len(arrNbClasses), len(arrLambPoisson)))
                matRes[:] = np.nan
                matStd = np.empty((len(arrR), len(arrNbClasses), len(arrLambPoisson)))
                matStd[:] = np.nan
                lab_arrNbClasses = {}
                lab_arrLambPoisson = {}

                for i_nbClasses, nbClasses in enumerate(arrNbClasses):
                    for i_lamb0_poisson, lamb0_poisson in enumerate(arrLambPoisson):
                        lamb0_poisson = np.round(lamb0_poisson, 5)
                        nbClasses = int(nbClasses)

                        lab_arrNbClasses[i_nbClasses] = str(nbClasses)
                        lab_arrLambPoisson[i_lamb0_poisson] = str(lamb0_poisson)

                        arrResR = [[] for _ in range(len(arrR))]

                        for DS in range(nbDS):
                            params = (folder, DS, nbClasses, num_obs, multivariate,
                                      overlap_voc, overlap_temp, perc_rand,
                                      voc_per_class, words_per_obs, theta0,
                                      lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                            try:
                                name_ds, observations, vocabulary_size = getData(params)
                                name_ds = name_ds.replace("_events.txt", "")
                            except Exception as e:
                                print(f"Data not found - {e}")
                                continue

                            for i_r, r in enumerate(arrR):

                                print(f"{model} - DS {DS} - lamb0_poisson = {lamb0_poisson} - nbClasses = {nbClasses} - r = {r}")
                                r = np.round(r, 2)

                                name_output = f"{name_ds}_r={r}" \
                                              f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                              f"_samplenum={sample_num}_particlenum={particle_num}"

                                try:
                                    DHP = read_particles(output_folder, name_output, get_clusters=False)
                                except Exception as e:
                                    print(f"Output not found - {e}")
                                    arrResR[i_r].append(np.nan)
                                    continue

                                selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                                clus_inf = selected_particle.docs2cluster_ID
                                clus_true = np.array(list(map(list, observations[:, 3])))[:, 0]

                                clus_inf = clus_inf[-num_NMI_last:]
                                clus_true = clus_true[-num_NMI_last:]

                                if len(clus_inf) != len(clus_true):
                                    arrResR[i_r].append(np.nan)
                                    continue

                                score = NMI(clus_true, clus_inf)

                                arrResR[i_r].append(score)
                                print(score)

                        arrResR = np.array(arrResR)
                        for i_r, r in enumerate(arrR):
                            if not np.isnan(arrResR[i_r]).all():
                                meanTabNMI = np.nanmean(arrResR[i_r])
                                stdTabNMI = nansem(arrResR[i_r])
                                matRes[i_r, i_nbClasses, i_lamb0_poisson] = meanTabNMI
                                matStd[i_r, i_nbClasses, i_lamb0_poisson] = stdTabNMI

                        scale=6
                        plt.figure(figsize=(1*scale, 1*scale))
                        for i_r, r in enumerate(arrR):
                            lab_x = [str(lab_arrNbClasses[idx]) for idx in lab_arrNbClasses]
                            lab_y = [str(lab_arrLambPoisson[idx]) for idx in lab_arrLambPoisson]


                            sns.heatmap(np.round(matRes[i_r], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=True,
                                        cbar_kws={"label":"NMI", "shrink": 0.6}, vmin=0, vmax=1)
                            plt.title(f"r = {r}")
                            plt.gca().invert_yaxis()
                            for ix in range(len(lab_arrNbClasses)):
                                for iy in range(len(lab_arrLambPoisson)):
                                    col = "white"
                                    if np.round(matRes[i_r, ix, iy], 2)<0.47: col="k"
                                    plt.text(ix+0.5, iy+0.2, fr"$\pm${np.round(matStd[i_r, ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("# classes")
                            plt.ylabel(r"$\lambda_0$")
                        plt.tight_layout()
                        plt.savefig(results_folder+"heatmap.pdf")
                        plt.close()

        # Words per obs vs overlap voc
        def XP3(folder, output_folder):
            folder += "XP3/"
            output_folder_based = output_folder + "XP3/"

            for model in ["MPDHP", "PDHP", "PDP", ]:
                output_folder = output_folder_based + model + "/"
                results_folder = output_folder.replace("output/", "results/")
                ensureFolder(results_folder)

                arr_words_per_obs = [2, 5, 8, 10, 15, 20, 25, 30]
                arr_overlap_voc = np.linspace(0, 1, 6)
                arrR = [1.]


                matRes = np.empty((len(arrR), len(arr_words_per_obs), len(arr_overlap_voc)))
                matRes[:] = np.nan
                matStd = np.empty((len(arrR), len(arr_words_per_obs), len(arr_overlap_voc)))
                matStd[:] = np.nan
                lab_arr_words_per_obs = {}
                lab_arr_overlap_voc = {}

                for i_words_per_obs,words_per_obs in enumerate(arr_words_per_obs):
                    for i_overlap_voc,overlap_voc in enumerate(arr_overlap_voc):
                        words_per_obs = int(words_per_obs)
                        overlap_voc = np.round(overlap_voc, 3)

                        lab_arr_words_per_obs[i_words_per_obs] = str(words_per_obs)
                        lab_arr_overlap_voc[i_overlap_voc] = str(overlap_voc)

                        arrResR = [[] for _ in range(len(arrR))]

                        for DS in range(nbDS):
                            params = (folder, DS, nbClasses, num_obs, multivariate,
                                      overlap_voc, overlap_temp, perc_rand,
                                      voc_per_class, words_per_obs, theta0,
                                      lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                            try:
                                name_ds, observations, vocabulary_size = getData(params)
                                name_ds = name_ds.replace("_events.txt", "")
                            except Exception as e:
                                print(f"Data not found - {e}")
                                continue

                            for i_r, r in enumerate(arrR):
                                print(f"{model} - DS {DS} - words per obs = {words_per_obs} - overlap_voc = {overlap_voc} - r = {r}")
                                r = np.round(r, 2)

                                name_output = f"{name_ds}_r={r}" \
                                              f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                              f"_samplenum={sample_num}_particlenum={particle_num}"

                                try:
                                    DHP = read_particles(output_folder, name_output, get_clusters=False)
                                except Exception as e:
                                    print(f"Output not found - {e}")
                                    arrResR[i_r].append(np.nan)
                                    continue

                                selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                                clus_inf = selected_particle.docs2cluster_ID
                                clus_true = np.array(list(map(list, observations[:, 3])))[:, 0]

                                clus_inf = clus_inf[-num_NMI_last:]
                                clus_true = clus_true[-num_NMI_last:]

                                if len(clus_inf) != len(clus_true):
                                    arrResR[i_r].append(np.nan)
                                    continue

                                score = NMI(clus_true, clus_inf)

                                arrResR[i_r].append(score)
                                print(score)

                        arrResR = np.array(arrResR)
                        for i_r, r in enumerate(arrR):
                            if not np.isnan(arrResR[i_r]).all():
                                meanTabNMI = np.nanmean(arrResR[i_r])
                                stdTabNMI = nansem(arrResR[i_r])
                                matRes[i_r, i_words_per_obs, i_overlap_voc] = meanTabNMI
                                matStd[i_r, i_words_per_obs, i_overlap_voc] = stdTabNMI

                        scale=6
                        plt.figure(figsize=(1*scale, 1*scale))
                        for i_r, r in enumerate(arrR):
                            lab_x = [str(lab_arr_words_per_obs[idx]) for idx in lab_arr_words_per_obs]
                            lab_y = [str(lab_arr_overlap_voc[idx]) for idx in lab_arr_overlap_voc]


                            sns.heatmap(np.round(matRes[i_r], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=True,
                                        cbar_kws={"label":"NMI", "shrink": 0.6}, vmin=0, vmax=1)
                            plt.title(f"r = {r}")

                            plt.gca().invert_yaxis()
                            for ix in range(len(lab_arr_words_per_obs)):
                                for iy in range(len(lab_arr_overlap_voc)):
                                    col = "white"
                                    if np.round(matRes[i_r, ix, iy], 2)<0.47: col="k"
                                    plt.text(ix+0.5, iy+0.2, fr"$\pm${np.round(matStd[i_r, ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("# words per event")
                            plt.ylabel("Textual overlap")
                        plt.tight_layout()
                        plt.savefig(results_folder+"heatmap.pdf")
                        plt.close()

        # Perc decorr vs r
        def XP4(folder, output_folder):
            folder += "XP4/"
            output_folder_based = output_folder + "XP4/"

            for model in ["MPDHP", "PDHP", "PDP", ]:
                output_folder = output_folder_based + model + "/"
                results_folder = output_folder.replace("output/", "results/")
                ensureFolder(results_folder)

                arr_perc_rand = np.linspace(0, 1, 6)
                arrR = np.linspace(0, 7, 15)

                matRes = np.empty((3, len(arrR), len(arr_perc_rand)))  # 3 = txt, temp, diff
                matRes[:] = np.nan
                matStd = np.empty((3, len(arrR), len(arr_perc_rand)))
                matStd[:] = np.nan
                lab_arr_perc_rand = {}

                for words_per_obs in [5, 10, 50]:
                    for i_perc_rand,perc_rand in enumerate(arr_perc_rand):
                        perc_rand = np.round(perc_rand, 2)

                        lab_arr_perc_rand[i_perc_rand] = str(perc_rand)

                        arrResR_txt = [[] for _ in range(len(arrR))]
                        arrResR_tmp = [[] for _ in range(len(arrR))]
                        arrResR_diff = [[] for _ in range(len(arrR))]

                        for i_r, r in enumerate(arrR):
                            r = np.round(r, 2)

                            for DS in range(nbDS):
                                params = (folder, DS, nbClasses, num_obs, multivariate,
                                          overlap_voc, overlap_temp, perc_rand,
                                          voc_per_class, words_per_obs, theta0,
                                          lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                                try:
                                    name_ds, observations, vocabulary_size = getData(params)
                                    name_ds = name_ds.replace("_events.txt", "")
                                except Exception as e:
                                    print(f"Data not found - {e}")
                                    continue

                                print(f"DS {DS} - perc rand = {perc_rand} - r = {r}")

                                name_output = f"{name_ds}_r={r}" \
                                              f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                              f"_samplenum={sample_num}_particlenum={particle_num}"

                                try:
                                    DHP = read_particles(output_folder, name_output, get_clusters=False)
                                except Exception as e:
                                    print(f"Output not found - {e}")
                                    arrResR_txt[i_r].append(np.nan)
                                    arrResR_tmp[i_r].append(np.nan)
                                    arrResR_diff[i_r].append(np.nan)
                                    continue

                                selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                                clus_inf = selected_particle.docs2cluster_ID
                                clus_true_txt = np.array(list(map(list, observations[:, 3])))[:, 0]
                                clus_true_tmp = np.array(list(map(list, observations[:, 3])))[:, 1]

                                clus_inf = clus_inf[-num_NMI_last:]
                                clus_true_txt = clus_true_txt[-num_NMI_last:]
                                clus_true_tmp = clus_true_tmp[-num_NMI_last:]


                                if len(clus_inf) != len(clus_true_txt):
                                    arrResR_txt[i_r].append(np.nan)
                                    arrResR_tmp[i_r].append(np.nan)
                                    arrResR_diff[i_r].append(np.nan)
                                    continue

                                score_txt = NMI(clus_true_txt, clus_inf)
                                score_tmp = NMI(clus_true_tmp, clus_inf)

                                arrResR_txt[i_r].append(score_txt)
                                arrResR_tmp[i_r].append(score_tmp)
                                arrResR_diff[i_r].append(score_txt-score_tmp)
                                print(score_txt-score_tmp)


                            if not np.isnan(arrResR_txt[i_r]).all():
                                meanTabNMI = np.nanmean(arrResR_txt[i_r])
                                stdTabNMI = nansem(arrResR_txt[i_r])
                                matRes[0, i_r, i_perc_rand] = meanTabNMI
                                matStd[0, i_r, i_perc_rand] = stdTabNMI
                            if not np.isnan(arrResR_tmp[i_r]).all():
                                meanTabNMI = np.nanmean(arrResR_tmp[i_r])
                                stdTabNMI = nansem(arrResR_tmp[i_r])
                                matRes[1, i_r, i_perc_rand] = meanTabNMI
                                matStd[1, i_r, i_perc_rand] = stdTabNMI
                            if not np.isnan(arrResR_diff[i_r]).all():
                                meanTabNMI = np.nanmean(arrResR_diff[i_r])
                                stdTabNMI = nansem(arrResR_diff[i_r])
                                matRes[2, i_r, i_perc_rand] = meanTabNMI
                                matStd[2, i_r, i_perc_rand] = stdTabNMI


                            lab_x = np.round(arrR, 2)
                            lab_y = [str(lab_arr_perc_rand[idx]) for idx in lab_arr_perc_rand]

                            scale=6
                            plt.figure(figsize=(3*scale, 1*scale))

                            plt.subplot(1,3,1)
                            sns.heatmap(np.round(matRes[0], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=False,
                                        cbar_kws={"label":r"NMI$_{text}$", "shrink": 0.6}, vmin=0, vmax=1)
                            plt.gca().invert_yaxis()
                            for ix in range(len(arrR)):
                                for iy in range(len(lab_arr_perc_rand)):
                                    col = "white"
                                    if np.round(matRes[0, ix, iy], 2)<0.47: col="k"
                                    pass
                                    #plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[0, ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("r")
                            plt.ylabel("Percentage decorrelated")


                            plt.subplot(1,3,2)
                            sns.heatmap(np.round(matRes[1], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=False,
                                        cbar_kws={"label":r"NMI$_{temp}$", "shrink": 0.6}, vmin=0, vmax=1)
                            plt.gca().invert_yaxis()
                            for ix in range(len(arrR)):
                                for iy in range(len(lab_arr_perc_rand)):
                                    col = "white"
                                    if np.round(matRes[1, ix, iy], 2)<0.47: col="k"
                                    pass
                                    #plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[1, ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("r")
                            plt.ylabel("Percentage decorrelated")

                            plt.subplot(1,3,3)
                            sns.heatmap(np.round(matRes[2], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="PuOr", square=True, annot=False,
                                        cbar_kws={"label":r"NMI$_{text}$-NMI$_{temp}$", "shrink": 0.6}, vmin=-1, vmax=1)
                            plt.gca().invert_yaxis()
                            for ix in range(len(arrR)):
                                for iy in range(len(lab_arr_perc_rand)):
                                    col = "white"
                                    if np.round(matRes[2, ix, iy], 2)<0.47: col="k"
                                    pass
                                    #plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[2, ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("r")
                            plt.ylabel("Percentage decorrelated")


                            plt.tight_layout()
                            plt.savefig(results_folder+f"heatmap_{words_per_obs}.pdf")
                            plt.close()

        # Univariate
        def XP5(folder, output_folder):
            folder += "XP5/"
            output_folder_based = output_folder + "XP5/"

            for model in ["MPDHP", "PDHP", "PDP", ]:
                output_folder = output_folder_based + model + "/"
                results_folder = output_folder.replace("output/", "results/")
                ensureFolder(results_folder)

                overlaps_voc = np.linspace(0, 1, 6)
                overlaps_temp = [0.]
                multivariate = False
                lamb0_classes = 0.15  # Slightly higher to avoid gaps in Hawkes process


                matRes = np.empty((len(arrR), len(overlaps_voc), len(overlaps_temp)))
                matRes[:] = np.nan
                matStd = np.empty((len(arrR), len(overlaps_voc), len(overlaps_temp)))
                matStd[:] = np.nan
                lab_overlap_voc = {}
                lab_overlap_temp = {}

                r_ref = 1.
                index_req1 = None
                for i_r, r in enumerate(arrR):
                    if r<r_ref+0.01 and r>r_ref-0.01: index_req1 = i_r

                for i_overlap_voc,overlap_voc in enumerate(overlaps_voc):
                    for i_overlap_temp,overlap_temp in enumerate(overlaps_temp):
                        overlap_voc = np.round(overlap_voc, 1)
                        overlap_temp = np.round(overlap_temp, 1)

                        lab_overlap_voc[i_overlap_voc] = str(overlap_voc)
                        lab_overlap_temp[i_overlap_temp] = str(overlap_temp)

                        arrResR = [[] for _ in range(len(arrR))]

                        for DS in range(nbDS):
                            params = (folder, DS, nbClasses, num_obs, multivariate,
                                      overlap_voc, overlap_temp, perc_rand,
                                      voc_per_class, words_per_obs, theta0,
                                      lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                            try:
                                name_ds, observations, vocabulary_size = getData(params)
                                name_ds = name_ds.replace("_events.txt", "")
                            except Exception as e:
                                print(f"Data not found - {e}")
                                continue

                            for i_r, r in enumerate(arrR):
                                print(f"DS {DS} - Univariate - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                                r = np.round(r, 2)

                                name_output = f"{name_ds}_r={r}" \
                                              f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                              f"_samplenum={sample_num}_particlenum={particle_num}"

                                try:
                                    DHP = read_particles(output_folder, name_output, get_clusters=False)
                                except Exception as e:
                                    print(f"Output not found - {e}")
                                    arrResR[i_r].append(np.nan)
                                    continue

                                selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                                clus_inf = selected_particle.docs2cluster_ID
                                clus_true = np.array(list(map(list, observations[:, 3])))[:, 0]

                                clus_inf = clus_inf[-num_NMI_last:]
                                clus_true = clus_true[-num_NMI_last:]

                                if len(clus_inf) != len(clus_true):
                                    arrResR[i_r].append(np.nan)
                                    continue

                                score = NMI(clus_true, clus_inf)

                                arrResR[i_r].append(score)
                                print(score)

                        arrResR = np.array(arrResR)
                        for i_r, r in enumerate(arrR):
                            if not np.isnan(arrResR[i_r]).all():
                                if i_r != index_req1:
                                    arrResR[i_r] -= arrResR[index_req1]
                                meanTabNMI = np.nanmean(arrResR[i_r])
                                stdTabNMI = nansem(arrResR[i_r])
                                matRes[i_r, i_overlap_voc, i_overlap_temp] = meanTabNMI
                                matStd[i_r, i_overlap_voc, i_overlap_temp] = stdTabNMI

                        scale=6
                        plt.figure(figsize=(4*scale, 1*scale))
                        for i_r, r in enumerate(arrR):

                            plt.subplot(1, 4, i_r+1)
                            lab_x = [str(lab_overlap_voc[idx]) for idx in lab_overlap_voc]
                            lab_y = [str(lab_overlap_temp[idx]) for idx in lab_overlap_temp]

                            if i_r == index_req1:
                                sns.heatmap(np.round(matRes[i_r], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=True,
                                            cbar_kws={"label":"NMI", "shrink": 0.6}, vmin=0, vmax=1)
                                plt.title(f"(reference) r = {r}")
                            else:
                                sns.heatmap(np.round(matRes[i_r], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="RdBu_r", square=True, annot=True,
                                            cbar_kws={"label":r"$\Delta$ NMI", "shrink": 0.6}, vmin=-1, vmax=1)
                                plt.title(f"r = {r}")


                            for ix in range(len(lab_overlap_voc)):
                                for iy in range(len(lab_overlap_temp)):
                                    col = "white"
                                    if np.round(matRes[i_r, ix, iy], 2)<0.47: col="k"
                                    plt.text(ix+0.5, iy+0.2, fr"$\pm${np.round(matStd[i_r, ix, iy], 2)}", ha="center", c=col, fontsize=7)

                            plt.gca().invert_yaxis()
                            plt.xlabel("Textual overlap")
                            plt.ylabel("Temporal overlap")
                        plt.tight_layout()
                        plt.savefig(results_folder+"heatmap.pdf")
                        plt.close()

        # Num part vs num sample
        def XP6(folder, output_folder):
            folder += "XP6/"
            output_folder_based = output_folder + "XP6/"

            for model in ["MPDHP", "PDHP", "PDP", ]:
                output_folder = output_folder_based + model + "/"
                results_folder = output_folder.replace("output/", "results/")
                ensureFolder(results_folder)

                num_part = [1, 2, 4, 8, 12, 16, 20, 25]
                num_sample = np.logspace(1, 5, 5)
                arrR = [1.]

                matRes = np.empty((len(arrR), len(num_part), len(num_sample)))
                matRes[:] = np.nan
                matStd = np.empty((len(arrR), len(num_part), len(num_sample)))
                matStd[:] = np.nan
                lab_num_part = {}
                lab_num_sample = {}






                for i_particle_num,particle_num in enumerate(num_part):
                    for i_sample_num,sample_num in enumerate(num_sample):
                        sample_num = int(sample_num)
                        particle_num = int(particle_num)

                        lab_num_part[i_particle_num] = str(particle_num)
                        lab_num_sample[i_sample_num] = str(sample_num)

                        arrResR = [[] for _ in range(len(arrR))]
                        for DS in range(nbDS):

                            params = (folder, DS, nbClasses, num_obs, multivariate,
                                      overlap_voc, overlap_temp, perc_rand,
                                      voc_per_class, words_per_obs, theta0,
                                      lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                            try:
                                name_ds, observations, vocabulary_size = getData(params)
                                name_ds = name_ds.replace("_events.txt", "")
                            except Exception as e:
                                print(f"Data not found - {e}")
                                continue



                            for i_r, r in enumerate(arrR):
                                print(f"DS {DS} - particles = {particle_num} - sample num = {sample_num} - r = {r}")
                                r = np.round(r, 2)

                                name_output = f"{name_ds}_r={r}" \
                                              f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                              f"_samplenum={sample_num}_particlenum={particle_num}"

                                try:
                                    DHP = read_particles(output_folder, name_output, get_clusters=False)
                                except Exception as e:
                                    print(f"Output not found - {e}")
                                    arrResR[i_r].append(np.nan)
                                    continue

                                selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                                clus_inf = selected_particle.docs2cluster_ID
                                clus_true = np.array(list(map(list, observations[:, 3])))[:, 0]

                                clus_inf = clus_inf[-num_NMI_last:]
                                clus_true = clus_true[-num_NMI_last:]

                                if len(clus_inf) != len(clus_true):
                                    arrResR[i_r].append(np.nan)
                                    continue

                                score = NMI(clus_true, clus_inf)

                                arrResR[i_r].append(score)
                                print(score)


                        arrResR = np.array(arrResR)
                        for i_r, r in enumerate(arrR):
                            if not np.isnan(arrResR[i_r]).all():
                                meanTabNMI = np.nanmean(arrResR[i_r])
                                stdTabNMI = nansem(arrResR[i_r])
                                matRes[i_r, i_particle_num, i_sample_num] = meanTabNMI
                                matStd[i_r, i_particle_num, i_sample_num] = stdTabNMI

                        scale=7
                        plt.figure(figsize=(1*scale, 1*scale))
                        for i_r, r in enumerate(arrR):

                            plt.subplot(1, 1, i_r+1)
                            lab_x = [str(lab_num_part[idx]) for idx in lab_num_part]
                            lab_y = [str(lab_num_sample[idx]) for idx in lab_num_sample]

                            sns.heatmap(np.round(matRes[i_r], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=True,
                                        cbar_kws={"label":"NMI", "shrink": 0.6}, vmin=0, vmax=1)
                            plt.title(f"r = {r}")
                            plt.gca().invert_yaxis()
                            plt.xlabel("# particles")
                            plt.ylabel("# samples")

                            for ix in range(len(lab_num_part)):
                                for iy in range(len(lab_num_sample)):
                                    col = "white"
                                    if np.round(matRes[i_r, ix, iy], 2)<0.47: col="k"
                                    plt.text(ix+0.5, iy+0.2, fr"$\pm${np.round(matStd[i_r, ix, iy], 2)}", ha="center", c=col, fontsize=7)

                        plt.tight_layout()
                        plt.savefig(results_folder+"heatmap.pdf")
                        plt.close()

        # MPDHP vs PDHP vs UP vs CRP
        def XP7(folder, output_folder):
            folder_base = folder
            for multivariate in [True, False]:
                if multivariate:
                    strMult = "Multivariate_data"
                    folder = folder_base + "XP1/"
                    output_folder_based = output_folder + "XP1/"
                else:
                    strMult = "Univariate_data"
                    folder = folder_base + "XP5/"
                    output_folder_based = output_folder + "XP5/"


                results_folder = output_folder_based.replace("output/", "results/").replace("XP1/", "XP7/").replace("XP5/", "XP7/")
                ensureFolder(results_folder)

                overlaps_voc = np.linspace(0, 1, 6)
                overlaps_temp = [0.]
                arrModels = ["MPDHP", "PDHP", "PDP", "UP"]

                matRes = np.empty((len(overlaps_voc), len(arrModels)))
                matRes[:] = np.nan
                matStd = np.empty((len(overlaps_voc), len(arrModels)))
                matStd[:] = np.nan
                lab_overlap_voc = {}
                lab_model = {}

                for i_model, model in enumerate(arrModels):
                    if model !="UP":
                        output_folder_final = output_folder_based + model + "/"
                    else:
                        output_folder_final = output_folder_based + "MPDHP" + "/"
                    for i_overlap_voc, overlap_voc in enumerate(sorted(overlaps_voc)):
                        for i_overlap_temp, overlap_temp in enumerate(sorted(overlaps_temp)):
                            overlap_voc = np.round(overlap_voc, 2)
                            overlap_temp = np.round(overlap_temp, 2)

                            lab_overlap_voc[i_overlap_voc] = str(overlap_voc)
                            strlabmod = model
                            if model == "PDHP": strlabmod = "DHP"
                            if model == "PDP": strlabmod = "DP"
                            lab_model[i_model] = strlabmod


                            arrResR = []
                            for DS in range(nbDS):
                                params = (folder, DS, nbClasses, num_obs, multivariate,
                                          overlap_voc, overlap_temp, perc_rand,
                                          voc_per_class, words_per_obs, theta0,
                                          lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                                try:
                                    name_ds, observations, vocabulary_size = getData(params)
                                    name_ds = name_ds.replace("_events.txt", "")
                                except Exception as e:
                                    print(f"Data not found - {e}")
                                    continue

                                if model=="UP":
                                    r=0.
                                else:
                                    r=1.

                                print(f"{model} - DS {DS} - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                                r = np.round(r, 2)

                                name_output = f"{name_ds}_r={r}" \
                                              f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                              f"_samplenum={sample_num}_particlenum={particle_num}"

                                try:
                                    DHP = read_particles(output_folder_final, name_output, get_clusters=False)
                                except Exception as e:
                                    print(f"Output not found - {e}")
                                    arrResR.append(np.nan)
                                    continue

                                selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                                clus_inf = selected_particle.docs2cluster_ID
                                clus_true = np.array(list(map(list, observations[:, 3])))[:, 0]

                                clus_inf = clus_inf[-num_NMI_last:]
                                clus_true = clus_true[-num_NMI_last:]

                                if len(clus_inf) != len(clus_true):
                                    arrResR.append(np.nan)
                                    continue

                                score = NMI(clus_true, clus_inf)

                                arrResR.append(score)
                                print(score)

                            arrResR = np.array(arrResR)

                            if not np.isnan(arrResR).all():
                                meanTabNMI = np.nanmean(arrResR)
                                stdTabNMI = nansem(arrResR)
                                matRes[i_overlap_voc, i_model] = meanTabNMI
                                matStd[i_overlap_voc, i_model] = stdTabNMI

                        scale=6
                        plt.figure(figsize=(1*scale, 1*scale))

                        plt.subplot(1, 1, 1)
                        lab_x = [str(lab_overlap_voc[idx]) for idx in lab_overlap_voc]
                        lab_y = [str(lab_model[idx]) for idx in lab_model]

                        sns.heatmap(np.round(matRes, 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=True,
                                    cbar_kws={"label":"NMI", "shrink": 0.6}, vmin=0, vmax=1)

                        for ix in range(len(lab_overlap_voc)):
                            for iy in range(len(lab_model)):
                                col = "white"
                                if np.round(matRes[ix, iy], 2)<0.47: col="k"
                                plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)
                        plt.xlabel("Textual overlap")
                        plt.tight_layout()
                        plt.savefig(results_folder+f"heatmap_{strMult}_ErrNumEst.pdf")
                        plt.close()

        # Learning rate
        def XP8(folder, output_folder):
            pass


        if XP=="all":
            XP1(folder, output_folder)
            XP2(folder, output_folder)
            XP3(folder, output_folder)
            XP4(folder, output_folder)
            XP5(folder, output_folder)
            XP6(folder, output_folder)
        if XP=="1":
            XP1(folder, output_folder)
        if XP=="2":
            XP2(folder, output_folder)
        if XP=="3":
            XP3(folder, output_folder)
        if XP=="4":
            XP4(folder, output_folder)
        if XP=="5":
            XP5(folder, output_folder)
        if XP=="6":
            XP6(folder, output_folder)
        if XP=="7":
            XP7(folder, output_folder)
        if XP=="8":
            XP8(folder, output_folder)

    else:

        means = None
        sigs = None

        try:
            timescale = sys.argv[3]
            #theta0 = float(sys.argv[4])
            arrThetas = [0.01, 0.001]
        except:
            timescale = "d"
            arrThetas = [0.01]

        arrR = [1.]#, 0.5, 0., 1.5]
        for theta0 in arrThetas:
            lamb0_poisson = 0.01  # Set at ~2sigma
            if True:
                if timescale=="min":
                    lamb0_poisson /= 1
                    means = [10*(i) for i in range(9)]  # Until 90min
                    sigs = [5 for i in range(9)]
                elif timescale=="h":
                    lamb0_poisson /= 10
                    means = [120*(i) for i in range(5)]  # Until 600min
                    sigs = [60 for i in range(5)]
                elif timescale=="d":
                    lamb0_poisson /= 100
                    means = [60*24*(i) for i in range(7)]  # Until 86400min
                    sigs = [60*24/2 for i in range(7)]

                means = np.array(means)
                sigs = np.array(sigs)

                alpha0 = 0.5  # Uniform beta or Dirichlet prior

                sample_num = 100000
                particle_num = 8
                multivariate = True

                folder = "data/News/"
                output_folder = "output/News/"

                lang = XP
                name_ds = f"allNews.txt"
                results_folder = f"results/News/{lang}/{timescale}/{np.round(theta0, 4)}/"
                ensureFolder(results_folder+"Clusters/")

            for r in arrR:
                name_output = f"News_timescale={timescale}_theta0={np.round(theta0,3)}_lamb0={lamb0_poisson}_" \
                              f"r={np.round(r,1)}_multi={multivariate}_samples={sample_num}_parts={particle_num}"


                indexToWd = readObservations(folder, name_ds, output_folder, onlyreturnindexwds=True)
                observations, vocabulary_size, indexToWd = readObservations(folder, name_ds, output_folder)
                DHP = read_particles(output_folder, name_output)
                observations = observations[:len(DHP.particles[0].docs2cluster_ID)]

                print(f"-------- r={r} - theta0={theta0} - time={timescale} --------")

                namethres = ""
                thresSizeLower = 100
                thresSizeUpper = 10000  # "Trash" clusters /100.000 obs
                for (namethres, thresSizeLower, thresSizeUpper) in [("_all", 10, 100000), ("_mediumclus", 50, 10000), ("_bigclus", 500, 100000)]:
                    name_output_res = name_output+namethres
                    numClusPerMonth = 5

                    un, cnt = np.unique(DHP.particles[0].docs2cluster_ID, return_counts=True)
                    print(list(sorted(cnt, reverse=True)))
                    un = un[cnt>thresSizeLower]
                    cnt = cnt[cnt>thresSizeLower]
                    un = un[cnt<thresSizeUpper]
                    cnt = cnt[cnt<thresSizeUpper]
                    consClus = [u for _, u in sorted(zip(cnt, un), reverse=True)]
                    print(consClus)

                    if len(consClus)==0:
                        continue

                    DHP = fill_clusters(DHP, consClus)

                    print("Computing A and weigths")
                    plotAdjTrans(results_folder, name_output_res, DHP, indexToWd, observations, consClus)

                    A = np.load(results_folder+name_output_res+"_adjacency.npy")
                    transparency = np.load(results_folder+name_output_res+"_transparency.npy")
                    transparency_permonth = np.load(results_folder+name_output_res+"_transparency_permonth.npy")
                    with open(results_folder+name_output_res+"_clusToInd.pkl", 'rb') as f:
                        clusToInd = pickle.load(f)

                    # print("Computing metrics")
                    # metrics(A, transparency, DHP, clusToInd)
                    # pause()

                    print("Computing timeline")
                    plotTimeline(observations, results_folder, name_output_res, DHP, indexToWd, consClus, numClusPerMonth=10)
                    for name_norm, axesNorm in [("_normKernel", [2]), ("_normOutEdges", [0]), ("_normInEdges", [1]), ("_normAbs", [0,1,2])]:
                        print(f"Computing graphs ({name_norm})")
                        plotGraphGlobEveryMonth(observations, A, transparency, transparency_permonth,
                                                results_folder, name_output_res+name_norm, DHP, indexToWd, consClus, clusToInd, numClusPerMonth=10, axesNorm=axesNorm)
                        plotGraphGlobEveryMonth(observations, A, transparency, transparency_permonth**0,
                                                results_folder, name_output_res+name_norm+"_sansTransp", DHP, indexToWd, consClus, clusToInd, numClusPerMonth=10, axesNorm=axesNorm)
                        plotGraphGlobEveryMonth(observations, A**0, transparency, transparency_permonth,
                                                results_folder, name_output_res+name_norm+"_onlyTransp", DHP, indexToWd, consClus, clusToInd, numClusPerMonth=10, axesNorm=axesNorm)
                        plotGraphGlob(A, transparency, results_folder, name_output_res+name_norm, DHP, indexToWd, consClus[:10], clusToInd, axesNorm=axesNorm)
                        pass
                    print("Computing individual clusters")
                    plotIndividualClusters(A, transparency, results_folder, namethres, DHP, indexToWd, observations, consClus[:40], clusToInd)













