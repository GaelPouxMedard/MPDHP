import gzip
import pickle
import os

import numpy as np
from utils import *
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.stats import sem
from wordcloud import WordCloud
import multidict as multidict
import re
import matplotlib.patheffects as pe


def ensureFolder(folder):
    curfol = "./"
    for fol in folder.split("/")[:-1]:
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

def nansem(x):
    x = np.array(x)
    return sem(x[~np.isnan(x)])

def readObservations(folder, name_ds, output_folder):
    dataFile = folder+name_ds
    observations = []
    wdToIndex, index = {}, 0
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
                print("BROKEN ===")
                break

    indexToWd = {}
    with open(output_folder+name_ds.replace("_events.txt", "")+"_indexWords.txt", "r", encoding="utf-8") as f:
        for line in f:
            index, wd = line.replace("\n", "").split("\t")
            indexToWd[int(index)] = wd

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

def read_particles(folderOut, nameOut, time=-1, get_clusters=False, only_pop_clus=False):
    if time==-1: txtTime = "_final"
    else: txtTime = f"_obs={int(time)}"

    with gzip.open(folderOut+nameOut+txtTime+"_particles.pkl.gz", "rb") as f:
        DHP = pickle.load(f)

    popClus, cnt = np.unique(DHP.particles[0].docs2cluster_ID, return_counts=True)
    selected_clus = [clus for _, clus in sorted(zip(cnt, popClus), reverse=True)][:20]
    print(sorted(cnt, reverse=True))
    #pause()

    if get_clusters:
        dicFilesClus = {}
        for i in range(len(DHP.particles)):
            lg = len(DHP.particles[i].files_clusters)
            for clus_num, (clus_index, file_cluster) in enumerate(DHP.particles[i].files_clusters):
                if file_cluster not in dicFilesClus:
                    if (clus_index in selected_clus and only_pop_clus) or not only_pop_clus:
                        dicFilesClus[file_cluster] = read_clusters(file_cluster)

        for i in range(len(DHP.particles)):
            lg = len(DHP.particles[i].files_clusters)
            for clus_num, (clus_index, file_cluster) in enumerate(DHP.particles[i].files_clusters):
                if (clus_index in selected_clus and only_pop_clus) or not only_pop_clus:
                    clus = dicFilesClus[file_cluster]
                    DHP.particles[i].clusters[clus_index] = clus

    return DHP

def read_clusters(file_cluster):
    with gzip.open(file_cluster, "rb") as f:
        cluster = pickle.load(f)

    return cluster

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

def makeWordCloud(dictWordsFreq):
    #alice_mask = np.array(Image.open("alice_mask.png"))

    x, y = np.ogrid[:1000, :1000]
    mask = (x - 500) ** 2 + (y - 500) ** 2 > 500 ** 2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color="white", max_words=500, mask=mask, colormap="cividis")
    # generate word cloud
    wc.generate_from_frequencies(dictWordsFreq)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.tight_layout()
    plt.axis("off")


if __name__=="__main__":
    try:
        RW = sys.argv[1]
        XP = sys.argv[2]
    except:
        RW = "0"
        XP = "7"

    num_NMI_last = 5000000000  # ======================================================================================================
    norm_err = 0.2  # ======================================================================================================


    if RW=="0":
        if True:
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
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)
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
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)
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
                arr_overlap_voc = [0., 0.2, 0.4, 0.6, 0.8]
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
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)
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

                for words_per_obs in [20]:
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
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("r")
                            plt.ylabel("Percentage decorrelated")


                            plt.subplot(1,3,2)
                            sns.heatmap(np.round(matRes[1], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=False,
                                        cbar_kws={"label":r"NMI$_{temp}$", "shrink": 0.6}, vmin=0, vmax=1)
                            plt.gca().invert_yaxis()
                            for ix in range(len(arrR)):
                                for iy in range(len(lab_arr_perc_rand)):
                                    col = "white"
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)
                            plt.xlabel("r")
                            plt.ylabel("Percentage decorrelated")

                            plt.subplot(1,3,3)
                            sns.heatmap(np.round(matRes[2], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="PuOr", square=True, annot=False,
                                        cbar_kws={"label":r"NMI$_{text}$-NMI$_{temp}$", "shrink": 0.6}, vmin=-1, vmax=1)
                            plt.gca().invert_yaxis()
                            for ix in range(len(arrR)):
                                for iy in range(len(lab_arr_perc_rand)):
                                    col = "white"
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)
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
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)

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
                                    if np.round(matRes[ix, iy], 2)<0.45: col="k"
                                    plt.text(ix+0.5, iy+0.8, fr"$\pm${np.round(matStd[ix, iy], 2)}", ha="center", c=col, fontsize=7)

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
                                if np.round(matRes[ix, iy], 2)<0.45: col="k"
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
        lamb0_poisson = 0.01  # Set at ~2sigma

        means = None
        sigs = None

        try:
            timescale = sys.argv[3]
            theta0 = float(sys.argv[4])
        except:
            timescale = "min"
            theta0 = 0.1  # Has already been documented for RW in LDA like models, DHP, etc ~0.1, 0.01 ; here it's 10 to ease computing the overlap_voc

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

            alpha0 = 1.  # Uniform beta or Dirichlet prior

            arrR = [0., 0.5, 1., 1.5]
            sample_num = 20000  # Typically 5 active clusters, so 25*len(mean) parameters to infer using sample_num*len(mean) samples => ~sample_num/25 samples per float
            particle_num = 20  # Like 10 simultaneous runs
            multivariate = True

            folder = "data/Covid/"
            output_folder = "output/Covid/"

            lang = XP
            name_ds = f"COVID-19-events_{lang}.txt"
            results_folder = f"temp/MPDHP/results/Covid/{lang}/{timescale}/"
            ensureFolder(results_folder+"Clusters/")

        for r in arrR:
            r = 1.
            name_output = f"COVID-19-events_{lang}_timescale={timescale}_theta0={np.round(theta0,3)}_lamb0={lamb0_poisson}_" \
                          f"r={np.round(r,1)}_multi={multivariate}_samples={sample_num}_parts={particle_num}"

            observations, vocabulary_size, indexToWd = readObservations(folder, name_ds, output_folder)
            DHP = read_particles(output_folder, name_output, get_clusters=True, only_pop_clus=True)

            print()
            print(f"-------- {r} ---------")


            scale = 4
            for c in sorted(DHP.particles[0].clusters):
                if len(DHP.particles[0].clusters[c].alpha_final)==0:  # Not present for r=0
                    continue

                plt.figure(figsize=(1*scale, 3*scale))
                plt.subplot(3,1,1)
                plt.title(f"Cluster {c}")
                wds = {indexToWd[idx]: count for count, idx in reversed(sorted(zip(DHP.particles[0].clusters[c].word_distribution, list(range(len(DHP.particles[0].clusters[c].word_distribution)))))) if count!=0}
                makeWordCloud(wds)

                plt.subplot(3,1,2)

                lg = len(observations)
                consClus = list(DHP.particles[0].clusters[c].alpha_final.keys())

                active_timestamps = np.array(list(zip(DHP.particles[0].docs2cluster_ID[:lg], observations[:, 1])))
                active_timestamps = np.array([ats for ats in active_timestamps if ats[0] in consClus])
                unweighted_triggering_kernel = np.zeros((len(consClus), len(means)))
                div = np.zeros((len(consClus)))
                indToClus = {int(c): i for i,c in enumerate(consClus)}
                for i in range(1, len(active_timestamps)):
                    if active_timestamps[i,0]!=c: continue  # Influence on c only
                    active_timestamps_cons = active_timestamps[active_timestamps[:, 1]>active_timestamps[i,1]-np.max(means)-3*np.max(sigs)]
                    active_timestamps_cons = active_timestamps_cons[active_timestamps_cons[:, 1]<active_timestamps[i,1]]
                    timeseq = active_timestamps_cons[:, 1]
                    clusseq = active_timestamps_cons[:, 0]
                    time_intervals = timeseq[-1] - timeseq[timeseq<timeseq[-1]]
                    if len(time_intervals)!=0:
                        RBF = RBF_kernel(means, time_intervals, sigs)  # num_time_points, size_kernel
                        for (clus, trig) in zip(clusseq, RBF):
                            indclus = indToClus[int(clus)]
                            unweighted_triggering_kernel[indclus][0] += trig.dot(DHP.particles[0].clusters[c].alpha_final[clus])  # [0] bc on multiplie déjà
                            div[indclus] += 1

                unweighted_triggering_kernel = np.sum(unweighted_triggering_kernel, axis=-1)
                unweighted_triggering_kernel /= (div+1e-20)  # Only Siths deal in absolutes
                print(unweighted_triggering_kernel)
                unweighted_triggering_kernel/=(np.max(unweighted_triggering_kernel)+1e-20)

                dt = np.linspace(0, np.max(means)+3*np.max(sigs), 1000)
                trigger = RBF_kernel(means, dt, sigs)
                for influencer_clus in DHP.particles[0].clusters[c].alpha_final:
                    trigger_clus = trigger.dot(DHP.particles[0].clusters[c].alpha_final[influencer_clus])
                    alpha = unweighted_triggering_kernel[indToClus[influencer_clus]]
                    if alpha<1e-5: continue
                    plt.plot(dt, trigger_clus, "-", label=f"Cluster {influencer_clus}", alpha=alpha)
                plt.legend()


                plt.subplot(3, 1, 3)

                array_intensity = []
                array_times = []
                for t in np.linspace(np.min(active_timestamps[:, 1]), np.max(active_timestamps[:, 1]), 1000):
                    active_timestamps_cons = active_timestamps[active_timestamps[:, 1]>t-np.max(means)-3*np.max(sigs)]
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

                    array_times.append(timeseq[-1])
                    array_intensity.append(unweighted_triggering_kernel.dot(DHP.particles[0].clusters[c].alpha_final[clus]))


                array_intensity = np.array(array_intensity)
                for influencer_clus in indToClus:
                    plt.plot(array_times, array_intensity[:, indToClus[influencer_clus]], "-o", markersize=2, label=f"Cluster {influencer_clus}")
                plt.legend()



                plt.tight_layout()
                plt.savefig(results_folder+f"/Clusters/cluster_{c}.pdf")
                #plt.show()
                plt.close()