import gzip
import pickle
import os
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


def read_particles(folderOut, nameOut, time=-1):
    if time==-1: txtTime = "_final"
    else: txtTime = f"_obs={int(time)}"

    with gzip.open(folderOut+nameOut+txtTime+"_particles.pkl.gz", "rb") as f:
        DHP = pickle.load(f)

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


def plot_heatmap(mat, labx, laby, cbar_lab):
    sns.heatmap(mat, xticklabels=labx, yticklabels=laby, cmap="afmhot_r", square=True, annot=True,
                cbar_kws={"label":cbar_lab})



if __name__=="__main__":
    try:
        RW = sys.argv[1]
        XP = sys.argv[2]
    except:
        RW = "0"
        XP = "1"

    plotFolder = "results/"


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

            arrR = [1., 0., 0.5, 1.5]
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

            overlaps_voc = np.linspace(0, 1, 6)
            overlaps_temp = np.linspace(0, 1, 6)

            t = time.time()
            i = 0
            nbRunsTot = nbDS*len(overlaps_voc)*len(overlaps_temp)*len(arrR)

            matRes = np.zeros((len(arrR), len(overlaps_voc), len(overlaps_temp)))-1
            matStd = np.zeros((len(arrR), len(overlaps_voc), len(overlaps_temp)))
            lab_overlap_voc = {}
            lab_overlap_temp = {}
            for i_r, r in enumerate(arrR):
                if r < 0.4 or r > 1.1: continue  # =========================================== REMOVE MEEEEEEEEEEEEEEEEEEEEEEEEEE
                for i_overlap_voc, overlap_voc in enumerate(sorted(overlaps_voc)):
                    for i_overlap_temp, overlap_temp in enumerate(sorted(overlaps_temp)):
                        overlap_voc = np.round(overlap_voc, 2)
                        overlap_temp = np.round(overlap_temp, 2)

                        lab_overlap_voc[i_overlap_voc] = str(overlap_voc)
                        lab_overlap_temp[i_overlap_temp] = str(overlap_temp)

                        tabNMI = []
                        prevclusinf = None
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

                            print(f"DS {DS} - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                          f"_samplenum={sample_num}_particlenum={particle_num}"

                            try:
                                print(output_folder, name_output)
                                DHP = read_particles(output_folder, name_output)
                            except Exception as e:
                                print(f"Output not found - {e}")
                                continue

                            clus_true = np.array(list(map(list, observations[:, 3])))[:, 0]
                            selected_particle = sorted(DHP.particles, key=lambda x: x.weight, reverse=True)[0]
                            clus_inf = selected_particle.docs2cluster_ID

                            if prevclusinf is not None:
                                print(np.isclose(clus_true, prevclusinf).all())
                            prevclusinf = clus_true.copy()
                            print(clus_true[:20])
                            print(clus_inf[:20])
                            print()
                            score = NMI(clus_true, clus_inf)
                            tabNMI.append(score)
                            print(score)

                            i += 1
                            print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                                  f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

                        if len(tabNMI) != 0:
                            meanTabNMI = np.mean(tabNMI)
                            stdTabNMI = np.std(tabNMI)
                            matRes[i_r, i_overlap_voc, i_overlap_temp] = meanTabNMI
                            matStd[i_r, i_overlap_voc, i_overlap_temp] = stdTabNMI

        # nbClasses vs lamb0
        def XP2(folder, output_folder):
            folder += "XP2/"
            output_folder += "XP2/"

            arrNbClasses = list(range(2, 10))
            arrLambPoisson = np.logspace(-4, 1, 6)
            arrR = [1.]

            t = time.time()
            i = 0
            nbRunsTot = nbDS*len(arrNbClasses)*len(arrLambPoisson)*len(arrR)

            num_obs = 100000
            for DS in range(nbDS):
                for nbClasses in arrNbClasses:
                    for lamb0_poisson in arrLambPoisson:
                        lamb0_poisson = np.round(lamb0_poisson, 5)
                        nbClasses = int(nbClasses)

                        params = (folder, DS, nbClasses, num_obs, multivariate,
                                  overlap_voc, overlap_temp, perc_rand,
                                  voc_per_class, words_per_obs, theta0,
                                  lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                        success = generate(params)
                        if success==-1: continue
                        name_ds, observations, vocabulary_size = getData(params)
                        name_ds = name_ds.replace("_events.txt", "")

                        for r in arrR:
                            print(f"DS {DS} - lamb0_poisson = {lamb0_poisson} - nbClasses = {nbClasses} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                          f"_samplenum={sample_num}_particlenum={particle_num}"

                            run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r,
                                    theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                    printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate,
                                    eval_on_go=eval_on_go)

                            i += 1
                            print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                                  f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

        # Words per obs vs overlap voc
        def XP3(folder, output_folder):
            folder += "XP3/"
            output_folder += "XP3/"

            arr_words_per_obs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
            arr_overlap_voc = np.linspace(0, 1, 6)
            arrR = [1.]

            t = time.time()
            i = 0
            nbRunsTot = nbDS*len(arr_words_per_obs)*len(arr_overlap_voc)*len(arrR)

            for DS in range(nbDS):
                for words_per_obs in arr_words_per_obs:
                    for overlap_voc in arr_overlap_voc:
                        words_per_obs = int(words_per_obs)
                        overlap_voc = np.round(overlap_voc, 3)

                        params = (folder, DS, nbClasses, num_obs, multivariate,
                                  overlap_voc, overlap_temp, perc_rand,
                                  voc_per_class, words_per_obs, theta0,
                                  lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                        success = generate(params)
                        if success==-1: continue
                        name_ds, observations, vocabulary_size = getData(params)
                        name_ds = name_ds.replace("_events.txt", "")

                        for r in arrR:
                            print(f"DS {DS} - words per obs = {words_per_obs} - overlap_voc = {overlap_voc} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                          f"_samplenum={sample_num}_particlenum={particle_num}"

                            run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r,
                                    theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                    printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate,
                                    eval_on_go=eval_on_go)


                            i += 1
                            print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                                  f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

        # Perc decorr vs r
        def XP4(folder, output_folder):
            folder += "XP4/"
            output_folder += "XP4/"

            arr_perc_rand = np.linspace(0, 1, 6)
            arrR = np.linspace(0, 3, 16)

            t = time.time()
            i = 0
            nbRunsTot = nbDS*len(arr_perc_rand)*len(arrR)

            for DS in range(nbDS):
                for perc_rand in arr_perc_rand:
                    perc_rand = np.round(perc_rand, 2)

                    params = (folder, DS, nbClasses, num_obs, multivariate,
                              overlap_voc, overlap_temp, perc_rand,
                              voc_per_class, words_per_obs, theta0,
                              lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                    success = generate(params)
                    if success==-1: continue
                    name_ds, observations, vocabulary_size = getData(params)
                    name_ds = name_ds.replace("_events.txt", "")

                    for r in arrR:
                        print(f"DS {DS} - perc rand = {perc_rand} - r = {r}")
                        r = np.round(r, 2)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r,
                                theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate,
                                eval_on_go=eval_on_go)


                        i += 1
                        print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                              f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

        # Univariate
        def XP5(folder, output_folder):
            folder += "XP5/"
            output_folder += "XP5/"

            overlaps_voc = np.linspace(0, 1, 6)
            overlaps_temp = np.linspace(0, 1, 6)
            multivariate = False
            lamb0_classes = 0.15  # Slightly higher to avoid gaps in Hawkes process

            t = time.time()
            i = 0
            nbRunsTot = nbDS*len(overlaps_voc)*len(overlaps_temp)*len(arrR)

            for DS in range(nbDS):
                for overlap_voc in overlaps_voc:
                    for overlap_temp in overlaps_temp:
                        overlap_voc = np.round(overlap_voc, 1)
                        overlap_temp = np.round(overlap_temp, 1)

                        params = (folder, DS, nbClasses, num_obs, multivariate,
                                  overlap_voc, overlap_temp, perc_rand,
                                  voc_per_class, words_per_obs, theta0,
                                  lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                        success = generate(params)
                        if success==-1: continue
                        name_ds, observations, vocabulary_size = getData(params)
                        name_ds = name_ds.replace("_events.txt", "")

                        for r in arrR:
                            print(f"DS {DS} - Univariate - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                          f"_samplenum={sample_num}_particlenum={particle_num}"

                            run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r,
                                    theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                    printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate,
                                    eval_on_go=eval_on_go)

                            i += 1
                            print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                                  f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

        # Num part vs num sample
        def XP6(folder, output_folder):
            folder += "XP6/"
            output_folder += "XP6/"

            num_part = [1, 2, 4, 8, 12, 16, 20, 25]
            num_sample = np.logspace(1, 5, 5)
            arrR = [1.]

            t = time.time()
            i = 0
            nbRunsTot = nbDS*len(num_part)*len(num_sample)*len(arrR)

            for DS in range(nbDS):

                params = (folder, DS, nbClasses, num_obs, multivariate,
                          overlap_voc, overlap_temp, perc_rand,
                          voc_per_class, words_per_obs, theta0,
                          lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                success = generate(params)
                if success==-1: continue
                name_ds, observations, vocabulary_size = getData(params)
                name_ds = name_ds.replace("_events.txt", "")

                for particle_num in num_part:
                    for sample_num in num_sample:
                        sample_num = int(sample_num)
                        particle_num = int(particle_num)

                        for r in arrR:
                            print(f"DS {DS} - Univariate - particles = {particle_num} - sample num = {sample_num} - r = {r}")
                            r = np.round(r, 2)

                            name_output = f"{name_ds}_r={r}" \
                                          f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                          f"_samplenum={sample_num}_particlenum={particle_num}"

                            run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r,
                                    theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                    printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate,
                                    eval_on_go=eval_on_go)

                            i += 1
                            print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                                  f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")



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
