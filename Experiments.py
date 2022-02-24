import os
import numpy as np
import sys
from Generate_data import generate
from MPDHP import run_fit

np.random.seed(1111)

def ensureFolder(folder):
    curfol = "./"
    for fol in folder.split("/")[:-1]:
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

def readObservations(folder, name_ds, output_folder):
    ensureFolder(output_folder)
    dataFile = folder+name_ds+"_events.txt"
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
    with open(output_folder+name_ds+"_indexWords.txt", "w+", encoding="utf-8") as f:
        for wd in wdToIndex:
            f.write(f"{wdToIndex[wd]}\t{wd}\n")
    V = len(wdToIndex)
    return observations, V

def getData(params):
    (folder, DS, nbClasses, run_time, multivariate,
     overlap_voc, overlap_temp, perc_rand,
     voc_per_class, words_per_obs, theta0,
     lamb0_poisson, lamb0_classes, alpha0, means, sigs) = params

    name_ds = f"Obs_nbclasses={nbClasses}_lg={run_time}_overlapvoc={overlap_voc}_overlaptemp={overlap_temp}" \
              f"_percrandomizedclus={perc_rand}_vocperclass={voc_per_class}_wordsperevent={words_per_obs}_DS={DS}"

    observations, vocabulary_size = readObservations(folder, name_ds, output_folder)

    return name_ds, observations, vocabulary_size


RW = sys.argv[1]
XP = sys.argv[2]

if RW=="0":
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

    r = 1.
    nbDS = 10
    sample_num = 2000  # Typically 5 active clusters, so 25*5 parameters to infer using 2000*5 samples => ~80 samples per parameter
    particle_num = 10  # Like 10 simultaneous runs
    multivariate = True
    printRes = True

    folder = "data/Synth/"
    output_folder = "output/Synth/"


    # Overlap voc vs overlap temp
    def XP1(folder, output_folder):
        folder += "XP1/"
        output_folder += "XP1/"
        for DS in range(nbDS):
            for overlap_voc in np.linspace(0, 1, 11):
                for overlap_temp in np.linspace(0, 1, 11):
                    overlap_voc = np.round(overlap_voc, 1)
                    overlap_temp = np.round(overlap_temp, 1)

                    params = (folder, DS, nbClasses, num_obs, multivariate,
                              overlap_voc, overlap_temp, perc_rand,
                              voc_per_class, words_per_obs, theta0,
                              lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                    success = generate(params)
                    if success==-1: continue
                    name_ds, observations, vocabulary_size = getData(params)

                    for r in [1., 0., 0.5, 1.5]:
                        print(f"DS {DS} - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                        r = np.round(r, 1)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r, theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num, printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate)

                        sys.exit()

    # nbClasses vs lamb0
    def XP2(folder, output_folder):
        folder += "XP2/"
        output_folder += "XP2/"
        num_obs = 100000
        for DS in range(nbDS):
            for nbClasses in list(range(1, 10)):
                for lamb0_poisson in np.logspace(-4, 1, 11):
                    lamb0_poisson = np.round(lamb0_poisson, 1)
                    nbClasses = np.round(nbClasses, 1)

                    params = (folder, DS, nbClasses, num_obs, multivariate,
                              overlap_voc, overlap_temp, perc_rand,
                              voc_per_class, words_per_obs, theta0,
                              lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                    success = generate(params)
                    if success==-1: continue
                    name_ds, observations, vocabulary_size = getData(params)

                    for r in [1., 0., 0.5, 1.5]:
                        print(f"DS {DS} - lamb0_poisson = {lamb0_poisson} - nbClasses = {nbClasses} - r = {r}")
                        r = np.round(r, 1)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r, theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num, printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate)

    # Words per obs vs overlap voc
    def XP3(folder, output_folder):
        folder += "XP3/"
        output_folder += "XP3/"
        for DS in range(nbDS):
            for words_per_obs in list(range(1, 20)):
                for overlap_voc in np.linspace(0, 1, 11):
                    words_per_obs = np.round(words_per_obs, 0)
                    overlap_voc = np.round(overlap_voc, 1)

                    params = (folder, DS, nbClasses, num_obs, multivariate,
                              overlap_voc, overlap_temp, perc_rand,
                              voc_per_class, words_per_obs, theta0,
                              lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                    success = generate(params)
                    if success==-1: continue
                    name_ds, observations, vocabulary_size = getData(params)

                    for r in [1., 0., 0.5, 1.5]:
                        print(f"DS {DS} - words per obs = {words_per_obs} - overlap_voc = {overlap_voc} - r = {r}")
                        r = np.round(r, 1)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r, theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num, printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate)

    # Perc decorr vs r
    def XP4(folder, output_folder):
        folder += "XP4/"
        output_folder += "XP4/"
        for DS in range(nbDS):
            for perc_rand in np.linspace(0, 1, 11):
                perc_rand = np.round(perc_rand, 0)

                params = (folder, DS, nbClasses, num_obs, multivariate,
                          overlap_voc, overlap_temp, perc_rand,
                          voc_per_class, words_per_obs, theta0,
                          lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                success = generate(params)
                if success==-1: continue
                name_ds, observations, vocabulary_size = getData(params)

                for r in np.linspace(0, 3, 31):
                    print(f"DS {DS} - perc rand = {perc_rand} - r = {r}")
                    r = np.round(r, 1)

                    name_output = f"{name_ds}_r={r}" \
                                  f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                  f"_samplenum={sample_num}_particlenum={particle_num}"

                    run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r, theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num, printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate)

    # Univariate
    def XP5(folder, output_folder):
        folder += "XP5/"
        output_folder += "XP5/"
        multivariate = False
        for DS in range(nbDS):
            for overlap_voc in np.linspace(0, 1, 11):
                for overlap_temp in np.linspace(0, 1, 11):
                    overlap_voc = np.round(overlap_voc, 1)
                    overlap_temp = np.round(overlap_temp, 1)

                    params = (folder, DS, nbClasses, num_obs, multivariate,
                              overlap_voc, overlap_temp, perc_rand,
                              voc_per_class, words_per_obs, theta0,
                              lamb0_poisson, lamb0_classes, alpha0, means, sigs)

                    success = generate(params)
                    if success==-1: continue
                    name_ds, observations, vocabulary_size = getData(params)

                    for r in [1., 0., 0.5, 1.5]:
                        print(f"DS {DS} - Univariate - overlap voc = {overlap_voc} - overlap temp = {overlap_temp} - r = {r}")
                        r = np.round(r, 1)

                        name_output = f"{name_ds}_r={r}" \
                                      f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0_classes}" \
                                      f"_samplenum={sample_num}_particlenum={particle_num}"

                        run_fit(observations, output_folder, name_output, lamb0_poisson, means, sigs, r=r, theta0=theta0, alpha0=alpha0, sample_num=sample_num, particle_num=particle_num, printRes=printRes, vocabulary_size=vocabulary_size, multivariate=multivariate)


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

else:
    pass