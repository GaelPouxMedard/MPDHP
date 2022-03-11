import gzip
import pickle
import os

import numpy as np

from utils import *
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score as NMI

def ensureFolder(folder):
    curfol = "./"
    for fol in folder.split("/")[:-1]:
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

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
                clusTxt = int(float(l[2]))
                clusTmp = int(float(l[3]))
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

    indexToWd = {}
    with open(output_folder+name_ds.replace("_events.txt", "")+"_indexWords.txt", "r", encoding="utf-8") as f:
        for line in f:
            index, wd = line.replace("\n", "").split("\t")
            wdToIndex[index] = wd

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


def read_particles(folderOut, nameOut, time=-1, get_clusters=False):
    if time==-1: txtTime = "_final"
    else: txtTime = f"_obs={int(time)}"

    with gzip.open(folderOut+nameOut+txtTime+"_particles.pkl.gz", "rb") as f:
        DHP = pickle.load(f)

    if get_clusters:
        dicFilesClus = {}
        for i in range(len(DHP.particles)):
            lg = len(DHP.particles[i].files_clusters)
            for clus_num, (clus_index, file_cluster) in enumerate(DHP.particles[i].files_clusters):
                if file_cluster not in dicFilesClus:
                    dicFilesClus[file_cluster] = read_clusters(file_cluster)

        for i in range(len(DHP.particles)):
            lg = len(DHP.particles[i].files_clusters)
            for clus_num, (clus_index, file_cluster) in enumerate(DHP.particles[i].files_clusters):
                clus = dicFilesClus[file_cluster]
                DHP.particles[i].clusters[clus_index] = clus

    return DHP


def read_clusters(file_cluster):
    file_cluster = "temp/MPDHP/"+file_cluster  # ==================================================================================================== REMOVEEEEEEEEEEEEEEEEEEEEEEEE

    with gzip.open(file_cluster, "rb") as f:
        cluster = pickle.load(f)

    return cluster


if __name__=="__main__":
    try:
        RW = sys.argv[1]
        XP = sys.argv[2]
    except:
        RW = "0"
        XP = "1"

    num_NMI_last = 5000
    norm_err = 0.2
    nbDS = 2  # ======================================================================================================


    if RW=="0":
        if True:
            nbClasses = 2
            num_obs = 50000

            overlap_voc = 0.2  # Proportion of voc in common between a clusters and its direct neighbours
            overlap_temp = 0.2  # Overlap between the kernels of the simulating process
            perc_rand = 0.  # Percentage of events to which assign random textual cluster

            voc_per_class = 1000  # Number of words available for each cluster
            words_per_obs = 5  # Twitter or headlines typically have few named entities

            lamb0_poisson = 0.01  # Cannot be inferred
            lamb0_classes = 0.1  # Cannot be inferred
            theta0 = 10.  # Has already been documented for RW in LDA like models, DHP, etc ~0.01, 0.001 ; here it's 10 to ease computing the overlap_voc
            alpha0 = 1.  # Uniform beta or Dirichlet prior
            means = np.array([3, 5, 7, 11, 13])
            sigs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

            arrR = [0., 0.5, 1., 1.5]
            nbDS = 10
            sample_num = 2000  # Typically 5 active clusters, so 25*5 parameters to infer using 2000*5 samples => ~80 samples per parameter
            particle_num = 10  # Like 10 simultaneous runs
            multivariate = True
            printRes = True
            eval_on_go = True

            folder = "data/Synth/"
            output_folder = "output/Synth/"
            folder = "temp/MPDHP/data/Synth/"
            output_folder = "temp/MPDHP/output/Synth/"

        # Overlap voc vs overlap temp
        def XP1(folder, output_folder):
            folder += "XP1/"
            output_folder += "XP1/"
            results_folder = output_folder.replace("output/", "results/")
            ensureFolder(results_folder)

            overlaps_voc = np.array([0., 0.2, 0.4, 0.6, 0.8])
            overlaps_temp = np.array([0.2, 0.4, 0.6, 0.8])

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
                            print(f"DS {DS} - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
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

                            score = NMI(clus_true, clus_inf)

                            arrResR[i_r].append(score)
                            print(score)

                    arrResR = np.array(arrResR)
                    for i_r, r in enumerate(arrR):
                        if not np.isnan(arrResR[i_r]).all():
                            if i_r != index_req1:
                                arrResR[i_r] -= arrResR[index_req1]
                            meanTabNMI = np.nanmean(arrResR[i_r])
                            stdTabNMI = np.nanstd(arrResR[i_r])
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
                                plt.plot([ix+0.5-matStd[i_r, ix, iy]/(2*norm_err), ix+0.5+matStd[i_r, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")
                        plt.xlabel("Textual overlap")
                        plt.ylabel("Temporal overlap")
                    plt.tight_layout()
                    plt.savefig(results_folder+"heatmap.pdf")
                    plt.close()

        # nbClasses vs lamb0
        def XP2(folder, output_folder):
            folder += "XP2/"
            output_folder += "XP2/"
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

            num_obs = 100000
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

                            print(f"DS {DS} - lamb0_poisson = {lamb0_poisson} - nbClasses = {nbClasses} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
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

                            score = NMI(clus_true, clus_inf)

                            arrResR[i_r].append(score)
                            print(score)

                    arrResR = np.array(arrResR)
                    for i_r, r in enumerate(arrR):
                        if not np.isnan(arrResR[i_r]).all():
                            meanTabNMI = np.nanmean(arrResR[i_r])
                            stdTabNMI = np.nanstd(arrResR[i_r])
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
                                plt.plot([ix+0.5-matStd[i_r, ix, iy]/(2*norm_err), ix+0.5+matStd[i_r, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")
                        plt.xlabel("# classes")
                        plt.ylabel(r"$\lambda_0$")
                    plt.tight_layout()
                    plt.savefig(results_folder+"heatmap.pdf")
                    plt.close()

        # Words per obs vs overlap voc
        def XP3(folder, output_folder):
            folder += "XP3/"
            output_folder += "XP3/"
            results_folder = output_folder.replace("output/", "results/")
            ensureFolder(results_folder)

            arr_words_per_obs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
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
                            print(f"DS {DS} - words per obs = {words_per_obs} - overlap_voc = {overlap_voc} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
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

                            score = NMI(clus_true, clus_inf)

                            arrResR[i_r].append(score)
                            print(score)

                    arrResR = np.array(arrResR)
                    for i_r, r in enumerate(arrR):
                        if not np.isnan(arrResR[i_r]).all():
                            meanTabNMI = np.nanmean(arrResR[i_r])
                            stdTabNMI = np.nanstd(arrResR[i_r])
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
                                plt.plot([ix+0.5-matStd[i_r, ix, iy]/(2*norm_err), ix+0.5+matStd[i_r, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")
                        plt.xlabel("# words per event")
                        plt.ylabel("Textual overlap")
                    plt.tight_layout()
                    plt.savefig(results_folder+"heatmap.pdf")
                    plt.close()

        # Perc decorr vs r
        def XP4(folder, output_folder):
            folder += "XP4/"
            output_folder += "XP4/"
            results_folder = output_folder.replace("output/", "results/")
            ensureFolder(results_folder)

            arr_perc_rand = np.linspace(0, 1, 6)
            arrR = np.linspace(0, 3, 16)

            words_per_obs = 20  # 5, 10, 20

            matRes = np.empty((3, len(arrR), len(arr_perc_rand)))  # 3 = txt, temp, diff
            matRes[:] = np.nan
            matStd = np.empty((3, len(arrR), len(arr_perc_rand)))
            matStd[:] = np.nan
            lab_arr_perc_rand = {}



            for i_perc_rand,perc_rand in enumerate(arr_perc_rand):
                perc_rand = np.round(perc_rand, 2)

                lab_arr_perc_rand[i_perc_rand] = str(perc_rand)

                arrResR_txt = [[] for _ in range(len(arrR))]
                arrResR_tmp = [[] for _ in range(len(arrR))]
                arrResR_diff = [[] for _ in range(len(arrR))]

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
                        print(f"DS {DS} - perc rand = {perc_rand} - r = {r}")
                        r = np.round(r, 2)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
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

                        score_txt = NMI(clus_true_txt, clus_inf)
                        score_tmp = NMI(clus_true_tmp, clus_inf)

                        arrResR_txt[i_r].append(score_txt)
                        arrResR_tmp[i_r].append(score_tmp)
                        arrResR_diff[i_r].append(score_txt-score_tmp)
                        print(score_txt-score_tmp)


                        if not np.isnan(arrResR_txt[i_r]).all():
                            meanTabNMI = np.nanmean(arrResR_txt[i_r])
                            stdTabNMI = np.nanstd(arrResR_txt[i_r])
                            matRes[0, i_r, i_perc_rand] = meanTabNMI
                            matStd[0, i_r, i_perc_rand] = stdTabNMI
                        if not np.isnan(arrResR_tmp[i_r]).all():
                            meanTabNMI = np.nanmean(arrResR_tmp[i_r])
                            stdTabNMI = np.nanstd(arrResR_tmp[i_r])
                            matRes[1, i_r, i_perc_rand] = meanTabNMI
                            matStd[1, i_r, i_perc_rand] = stdTabNMI
                        if not np.isnan(arrResR_diff[i_r]).all():
                            meanTabNMI = np.nanmean(arrResR_diff[i_r])
                            stdTabNMI = np.nanstd(arrResR_diff[i_r])
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
                                plt.plot([ix+0.5-matStd[0, ix, iy]/(2*norm_err), ix+0.5+matStd[0, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")
                        plt.xlabel("r")
                        plt.ylabel("Percentage decorrelated")


                        plt.subplot(1,3,2)
                        sns.heatmap(np.round(matRes[1], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="afmhot_r", square=True, annot=False,
                                    cbar_kws={"label":r"NMI$_{temp}$", "shrink": 0.6}, vmin=0, vmax=1)
                        plt.gca().invert_yaxis()
                        for ix in range(len(arrR)):
                            for iy in range(len(lab_arr_perc_rand)):
                                plt.plot([ix+0.5-matStd[1, ix, iy]/(2*norm_err), ix+0.5+matStd[1, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")
                        plt.xlabel("r")
                        plt.ylabel("Percentage decorrelated")

                        plt.subplot(1,3,3)
                        sns.heatmap(np.round(matRes[2], 2).T, xticklabels=lab_x, yticklabels=lab_y, cmap="PuOr", square=True, annot=False,
                                    cbar_kws={"label":r"NMI$_{text}$-NMI$_{temp}$", "shrink": 0.6}, vmin=-1, vmax=1)
                        plt.gca().invert_yaxis()
                        for ix in range(len(arrR)):
                            for iy in range(len(lab_arr_perc_rand)):
                                plt.plot([ix+0.5-matStd[2, ix, iy]/(2*norm_err), ix+0.5+matStd[2, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")
                        plt.xlabel("r")
                        plt.ylabel("Percentage decorrelated")


                        plt.tight_layout()
                        plt.savefig(results_folder+"heatmap.pdf")
                        plt.close()

        # Univariate
        def XP5(folder, output_folder):
            folder += "XP5/"
            output_folder += "XP5/"
            results_folder = output_folder.replace("output/", "results/")
            ensureFolder(results_folder)

            overlaps_voc = np.linspace(0, 1, 6)
            overlaps_temp = np.linspace(0, 1, 6)
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
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
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

                            score = NMI(clus_true, clus_inf)

                            arrResR[i_r].append(score)
                            print(score)

                    arrResR = np.array(arrResR)
                    for i_r, r in enumerate(arrR):
                        if not np.isnan(arrResR[i_r]).all():
                            if i_r != index_req1:
                                arrResR[i_r] -= arrResR[index_req1]
                            meanTabNMI = np.nanmean(arrResR[i_r])
                            stdTabNMI = np.nanstd(arrResR[i_r])
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
                                plt.plot([ix+0.5-matStd[i_r, ix, iy]/(2*norm_err), ix+0.5+matStd[i_r, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")

                        plt.gca().invert_yaxis()
                        plt.xlabel("Textual overlap")
                        plt.ylabel("Temporal overlap")
                    plt.tight_layout()
                    plt.savefig(results_folder+"heatmap.pdf")
                    plt.close()

        # Num part vs num sample
        def XP6(folder, output_folder):
            folder += "XP6/"
            output_folder += "XP6/"
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
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
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

                            score = NMI(clus_true, clus_inf)

                            arrResR[i_r].append(score)
                            print(score)


                    arrResR = np.array(arrResR)
                    for i_r, r in enumerate(arrR):
                        if not np.isnan(arrResR[i_r]).all():
                            meanTabNMI = np.nanmean(arrResR[i_r])
                            stdTabNMI = np.nanstd(arrResR[i_r])
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
                                plt.plot([ix+0.5-matStd[i_r, ix, iy]/(2*norm_err), ix+0.5+matStd[i_r, ix, iy]/(2*norm_err)], [iy+0.2]*2, "-|", c="gray")

                    plt.tight_layout()
                    plt.savefig(results_folder+"heatmap.pdf")
                    plt.close()



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
