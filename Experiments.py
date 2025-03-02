import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import sys
from Generate_data import generate
from MPDHP import run_fit
import time

np.random.seed(1111)

def ensureFolder(folder):
    curfol = "./"
    for fol in folder.split("/")[:-1]:
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

def readObservations(folder, name_ds, output_folder):
    ensureFolder(output_folder)
    dataFile = folder+name_ds
    observations = []
    wdToIndex, index = {}, 0
    with open(dataFile, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.replace("\n", "").split("\t")
            timestamp = float(l[0])
            words = l[1].split(",")
            try:
                clusTxt = l[2]
                clusTmp = l[3]
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

            tup = (i, timestamp, (uniquewords, cntwords), (clusTxt, clusTmp))
            observations.append(tup)

            if i > 1000:
                pass
                # print("BROKEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEN")
                # break

    with open(output_folder+name_ds.replace("_events.txt", "")+"_indexWords.txt", "w+", encoding="utf-8") as f:
        for wd in wdToIndex:
            f.write(f"{wdToIndex[wd]}\t{wd}\n")
    V = len(wdToIndex)
    indexToWd = {idx: wd for wd, idx in wdToIndex.items()}
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


try:
    RW = sys.argv[1]
    XP = sys.argv[2]
except:
    RW = "2"
    XP = "1"


if RW=="0":
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
    multivariate = True

    arrR = [1., 0.]
    nbDS = 100
    sample_num = 2000  # Typically 5 active clusters, so 25*5 parameters to infer using 2000*5 samples => ~80 samples per parameter
    particle_num = 10  # Like 10 simultaneous runs
    multivariate_fit = True
    simple_DP = False
    printRes = True
    eval_on_go = True

    folder = "data/Synth/"
    output_folder = "output/Synth/"


    def XP_temp(folder, output_folder):
        folder += "XP_temp/"
        output_folder += "XP_temp/"

        print(output_folder)

        DS = 0
        r = 1.
        num_obs = 100
        multivariate = False
        overlap_voc = 0.5
        overlap_temp = 0.

        params = (folder, DS, nbClasses, num_obs, multivariate,
                  overlap_voc, overlap_temp, perc_rand,
                  voc_per_class, words_per_obs, theta0,
                  lamb0_poisson, lamb0_classes, alpha0, means, sigs)

        generate(params)
        name_ds, observations, vocabulary_size = getData(params)
        name_ds = name_ds.replace("_events.txt", "")

        print(f"DS {DS} - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
        r = np.round(r, 2)

        name_output = f"{name_ds}_r={r}" \
                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                      f"_samplenum={sample_num}_particlenum={particle_num}"

        run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r,
                theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,
                eval_on_go=eval_on_go)

    # Overlap voc vs overlap temp
    def XP1(folder, output_folder):
        folder += "XP1/"
        output_folder += "XP1/"

        overlaps_voc = np.linspace(0, 1, 6)
        overlaps_temp = np.linspace(0, 1, 6)

        t = time.time()
        i = 0
        nbRunsTot = nbDS*len(overlaps_voc)*len(overlaps_temp)*len(arrR)

        for DS in range(nbDS):
            for overlap_temp in overlaps_temp:
                overlap_temp = np.round(overlap_temp, 2)
                overlaps_voc = np.round(overlaps_voc, 2)
                params = (folder, DS, nbClasses, num_obs, multivariate,
                          overlaps_voc, overlap_temp, perc_rand,
                          voc_per_class, words_per_obs, theta0,
                          lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                success = generate(params)
                if success==-1:
                    continue

                for overlap_voc in overlaps_voc:
                    overlap_voc = np.round(overlap_voc, 2)

                    params = (folder, DS, nbClasses, num_obs, multivariate,
                              overlap_voc, overlap_temp, perc_rand,
                              voc_per_class, words_per_obs, theta0,
                              lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                    try:
                        name_ds, observations, vocabulary_size = getData(params)
                    except:
                        continue
                    name_ds = name_ds.replace("_events.txt", "")

                    for r in arrR:
                        print(f"DS {DS} - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                        r = np.round(r, 2)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        for (multivariate_fit, simple_DP) in [(True, False), (False, False), (False, True)]:
                            if multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "MPDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if not multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "PDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if simple_DP:
                                output_folder_model = output_folder + "PDP/"
                                lamb0_poisson_used = lamb0_poisson*100


                            if (not simple_DP) or (simple_DP and r>1-1e-5 and r<1+1e-5):
                                if (multivariate_fit) or (not multivariate_fit and r>1-1e-5 and r<1+1e-5):
                                    print(f"r = {r} - Multivariate={multivariate_fit} - Simple DP={simple_DP}")
                                    run_fit(observations, output_folder_model, name_output, lamb0_poisson_used, means, sigs, r=r,
                                            theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                            printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,
                                            eval_on_go=eval_on_go)

                        i += 1
                        print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                              f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

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

        for DS in range(nbDS):
            for nbClasses in arrNbClasses:

                lamb0_poisson = -1
                params = (folder, DS, nbClasses, num_obs, multivariate,
                          overlap_voc, overlap_temp, perc_rand,
                          voc_per_class, words_per_obs, theta0,
                          lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                success = generate(params)
                if success==-1: continue
                name_ds, observations, vocabulary_size = getData(params)
                name_ds = name_ds.replace("_events.txt", "")

                for lamb0_poisson in arrLambPoisson:
                    lamb0_poisson = np.round(lamb0_poisson, 5)
                    nbClasses = int(nbClasses)

                    for r in arrR:
                        print(f"DS {DS} - lamb0_poisson = {lamb0_poisson} - nbClasses = {nbClasses} - r = {r}")
                        r = np.round(r, 2)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"
                        for (multivariate_fit, simple_DP) in [(True, False), (False, False), (False, True)]:
                            if multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "MPDHP/"
                                lamb0_poisson_used = lamb0_poisson
                                if r<1e-5:
                                    continue
                            if not multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "PDHP/"
                                lamb0_poisson_used = lamb0_poisson
                                continue
                            if simple_DP:
                                output_folder_model = output_folder + "PDP/"
                                lamb0_poisson_used = lamb0_poisson*100
                                continue

                            if (not simple_DP) or (simple_DP and r>1-1e-5 and r<1+1e-5):
                                if (multivariate_fit) or (not multivariate_fit and r>1-1e-5 and r<1+1e-5):
                                    print(f"r = {r} - Multivariate={multivariate_fit} - Simple DP={simple_DP}")
                                    run_fit(observations, output_folder_model, name_output, lamb0_poisson_used, means, sigs, r=r,
                                            theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                            printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,
                                            eval_on_go=eval_on_go)

                        i += 1
                        print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                              f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

    # Words per obs vs overlap voc
    def XP3(folder, output_folder):
        folder += "XP3/"
        output_folder += "XP3/"

        arr_words_per_obs = [2, 5, 8, 10, 15, 20, 25, 30]
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
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        for (multivariate_fit, simple_DP) in [(True, False), (False, False), (False, True)]:
                            if multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "MPDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if not multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "PDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if simple_DP:
                                output_folder_model = output_folder + "PDP/"
                                lamb0_poisson_used = lamb0_poisson*100

                            if (not simple_DP) or (simple_DP and r>1-1e-5 and r<1+1e-5):
                                if (multivariate_fit) or (not multivariate_fit and r>1-1e-5 and r<1+1e-5):
                                    print(f"r = {r} - Multivariate={multivariate_fit} - Simple DP={simple_DP}")
                                    run_fit(observations, output_folder_model, name_output, lamb0_poisson_used, means, sigs, r=r,
                                            theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                            printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,
                                            eval_on_go=eval_on_go)


                        i += 1
                        print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                              f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

    # Perc decorr vs r
    def XP4(folder, output_folder):
        folder += "XP4/"
        output_folder += "XP4/"

        arr_perc_rand = np.linspace(0, 1, 6)
        arrWdsPerObs = [5, 10, 50]
        arrR = np.linspace(0, 7, 15)
        words_per_obs = 50

        t = time.time()
        i = 0
        nbRunsTot = nbDS*len(arr_perc_rand)*len(arrR)*len(arrWdsPerObs)

        for words_per_obs in arrWdsPerObs:
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
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        for (multivariate_fit, simple_DP) in [(True, False), (False, False), (False, True)]:
                            if multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "MPDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if not multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "PDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if simple_DP:
                                output_folder_model = output_folder + "PDP/"
                                lamb0_poisson_used = lamb0_poisson*100

                            if (not simple_DP) or (simple_DP and r>1-1e-5 and r<1+1e-5):
                                if (multivariate_fit) or (not multivariate_fit and r>1-1e-5 and r<1+1e-5):
                                    print(f"r = {r} - Multivariate={multivariate_fit} - Simple DP={simple_DP}")
                                    run_fit(observations, output_folder_model, name_output, lamb0_poisson_used, means, sigs, r=r,
                                            theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                            printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,
                                            eval_on_go=eval_on_go)


                        i += 1
                        print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
                              f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")

    # Univariate
    def XP5(folder, output_folder):
        folder += "XP5/"
        output_folder += "XP5/"

        overlaps_voc = np.linspace(0, 1, 6)
        overlaps_temp = [0.]
        multivariate = False
        lamb0_classes = 0.15  # Slightly higher to avoid gaps in Hawkes process

        t = time.time()
        i = 0
        nbRunsTot = nbDS*len(overlaps_voc)*len(overlaps_temp)*len(arrR)

        for DS in range(nbDS):
            for overlap_temp in overlaps_temp:
                overlap_temp = np.round(overlap_temp, 2)
                overlaps_voc = np.round(overlaps_voc, 2)
                params = (folder, DS, nbClasses, num_obs, multivariate,
                          overlaps_voc, overlap_temp, perc_rand,
                          voc_per_class, words_per_obs, theta0,
                          lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                success = generate(params)
                if success==-1:
                    continue

                for overlap_voc in overlaps_voc:
                    overlap_voc = np.round(overlap_voc, 2)

                    params = (folder, DS, nbClasses, num_obs, multivariate,
                              overlap_voc, overlap_temp, perc_rand,
                              voc_per_class, words_per_obs, theta0,
                              lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                    try:
                        name_ds, observations, vocabulary_size = getData(params)
                    except:
                        continue
                    name_ds = name_ds.replace("_events.txt", "")

                    for r in arrR:
                        print(f"DS {DS} - Univariate - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                        r = np.round(r, 2)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        for (multivariate_fit, simple_DP) in [(True, False), (False, False), (False, True)]:
                            if multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "MPDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if not multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "PDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if simple_DP:
                                output_folder_model = output_folder + "PDP/"
                                lamb0_poisson_used = lamb0_poisson*100

                            if (not simple_DP) or (simple_DP and r>1-1e-5 and r<1+1e-5):
                                if (multivariate_fit) or (not multivariate_fit and r>1-1e-5 and r<1+1e-5):
                                    print(f"r = {r} - Multivariate={multivariate_fit} - Simple DP={simple_DP}")
                                    run_fit(observations, output_folder_model, name_output, lamb0_poisson_used, means, sigs, r=r,
                                            theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                            printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,
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
                        print(f"DS {DS} - particles = {particle_num} - sample num = {sample_num} - r = {r}")
                        r = np.round(r, 2)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_poisson}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        for (multivariate_fit, simple_DP) in [(True, False), (False, False), (False, True)]:
                            if multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "MPDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if not multivariate_fit and not simple_DP:
                                output_folder_model = output_folder + "PDHP/"
                                lamb0_poisson_used = lamb0_poisson
                            if simple_DP:
                                output_folder_model = output_folder + "PDP/"
                                lamb0_poisson_used = lamb0_poisson*100

                            if (not simple_DP) or (simple_DP and r>1-1e-5 and r<1+1e-5):
                                if (multivariate_fit) or (not multivariate_fit and r>1-1e-5 and r<1+1e-5):
                                    print(f"r = {r} - Multivariate={multivariate_fit} - Simple DP={simple_DP}")
                                    run_fit(observations, output_folder_model, name_output, lamb0_poisson_used, means, sigs, r=r,
                                            theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num,
                                            printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,
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
    if XP=="tmp":
        XP_temp(folder, output_folder)

elif RW=="1":
    lamb0_poisson = 0.01  # Set at ~2sigma

    means = None
    sigs = None

    try:
        timescale = sys.argv[3]
        theta0 = float(sys.argv[4])
    except:
        timescale = "min"
        theta0 = 0.1  # Has already been documented for RW in LDA like models, DHP, etc ~0.1, 0.01 ; here it's 10 to ease computing the overlap_voc

    if timescale=="min":
        lamb0_poisson /= 1
        means = [10*(i) for i in range(9)]  # Until 90min
        sigs = [5 for i in range(9)]
    elif timescale=="h":
        lamb0_poisson /= 10
        lamb0_poisson = 0.1
        means = [120*(i) for i in range(5)]  # Until 600min
        sigs = [60 for i in range(5)]
    elif timescale=="d":
        lamb0_poisson /= 100
        means = [60*24*(i) for i in range(7)]  # Until 86400min
        sigs = [60*24/2 for i in range(7)]

    means = np.array(means)
    sigs = np.array(sigs)

    alpha0 = 1.  # Uniform beta or Dirichlet prior

    arrR = [1., 0.5, 0., 1.5]
    sample_num = 20000  # Typically 5 active clusters, so 25*len(mean) parameters to infer using sample_num*len(mean) samples => ~sample_num/25 samples per float
    particle_num = 20  # Like 10 simultaneous runs
    multivariate_fit = True
    simple_DP = False
    printRes = True
    eval_on_go = False

    folder = "data/Covid/"
    output_folder = "output/Covid/"
    lg = XP
    name_ds = f"COVID-19-events_{lg}.txt"

    t = time.time()
    i = 0
    nbRunsTot = len(arrR)

    for r in arrR:
        name_output = f"COVID-19-events_{lg}_timescale={timescale}_theta0={np.round(theta0,3)}_lamb0={lamb0_poisson}_" \
                      f"r={np.round(r,1)}_multi={multivariate_fit}_samples={sample_num}_parts={particle_num}"

        # import pprofile
        # profiler = pprofile.Profile()
        # with profiler:

        observations, vocabulary_size, indexToWd = readObservations(folder, name_ds, output_folder)
        DHP = run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r, theta0=theta0, alpha0=alpha0,
                sample_num=sample_num, particle_num=particle_num, printRes=printRes,
                vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,  eval_on_go=eval_on_go, indexToWd=indexToWd)


        # profiler.print_stats()
        # profiler.dump_stats("Benchmark.txt")
        # pause()

        i += 1
        print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
              f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")


        for c in DHP.particles[0].active_clusters:
            wds = [idx for _, idx in reversed(sorted(zip(DHP.particles[0].clusters[c].word_distribution, list(range(len(DHP.particles[0].clusters[c].word_distribution))))))]
            print([indexToWd[idx] for idx in wds][:10],
                len(DHP.particles[0].clusters[c].word_distribution.nonzero()[0]), vocabulary_size)

elif RW=="2":

    lamb0_poisson = 0.01  # Set at ~2sigma

    means = None
    sigs = None

    try:
        timescale = sys.argv[3]
        theta0 = float(sys.argv[4])
    except:
        timescale = "h"
        theta0 = 0.1  # Has already been documented for RW in LDA like models, DHP, etc ~0.1, 0.01 ; here it's 10 to ease computing the overlap_voc

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

    alpha0 = 2.  # Beta or Dirichlet prior

    arrR = [1., 0.5, 0., 1.5]
    sample_num = 100000  # Typically 15 active clusters, so 15*15*len(mean)~1125 parameters to infer using sample_num*len(mean)~2000000 samples => sample_num/(15²*len(mean))~1777 samples per float
    particle_num = 8  # Like 8 simultaneous runs ==========================================
    multivariate_fit = True
    simple_DP = False
    printRes = True
    eval_on_go = False

    folder = "data/News/"
    output_folder = "output/News/"
    lg = XP
    name_ds = f"allNews.txt"

    t = time.time()
    i = 0
    nbRunsTot = len(arrR)

    for r in arrR:
        name_output = f"News_timescale={timescale}_theta0={np.round(theta0,3)}_lamb0={lamb0_poisson}_" \
                      f"r={np.round(r,1)}_multi={multivariate_fit}_samples={sample_num}_parts={particle_num}"


        observations, vocabulary_size, indexToWd = readObservations(folder, name_ds, output_folder)

        # import pprofile
        # profiler = pprofile.Profile()
        # with profiler:

        DHP = run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r, theta0=theta0, alpha0=alpha0,
                      sample_num=sample_num, particle_num=particle_num, printRes=printRes,
                      vocabulary_size=vocabulary_size, multivariate=multivariate_fit, simple_DP=simple_DP,  eval_on_go=eval_on_go, indexToWd=indexToWd)

        # profiler.print_stats()
        # profiler.dump_stats("Benchmark.txt")
        # pause()

        i += 1
        print(f"------------------------- r={r} - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - "
              f"ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")


        for c in DHP.particles[0].active_clusters:
            wds = [idx for _, idx in reversed(sorted(zip(DHP.particles[0].clusters[c].word_distribution, list(range(len(DHP.particles[0].clusters[c].word_distribution))))))]
            print([indexToWd[idx] for idx in wds][:10],
                  len(DHP.particles[0].clusters[c].word_distribution.nonzero()[0]), vocabulary_size)

