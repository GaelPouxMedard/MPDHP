import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy

from tick.hawkes import (SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp, HawkesEM)
from tick.hawkes import SimuPoissonProcess
from tick.base import TimeFunction
from tick.plot import plot_hawkes_kernels
import sys
import os
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'


def ensureFolder(folder):
    curfol = "./"
    for fol in folder.split("/"):
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

def gaussian(x, mu, sig):
    return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))/(2 * np.pi * np.power(sig, 2.)) ** 0.5

def kernel(dt, means, sigs, alpha):
    k = gaussian(dt[:, None], means[None, :], sigs[None, :]).dot(alpha)
    return k

def simulHawkes(lamb0_poisson, lamb0_classes, alpha, means, sigs, num_obs=1000):
    maxdt = max(means)+3*max(sigs)
    nbClasses = len(alpha)
    # Definition kernels
    emptyKer = HawkesKernelTimeFunc(TimeFunction(([0, 1], [0, 0]), inter_mode=TimeFunction.InterConstRight))
    kernels = [[copy(emptyKer) for _ in range(nbClasses)] for _ in range(nbClasses)]

    for c in range(nbClasses):
        for c2 in range(nbClasses):
            #if c!=c2: continue  # If uncommented: univariate Hawkes process
            t_values = np.linspace(0, maxdt, 100)
            y_values = kernel(t_values, means, sigs, alpha[c,c2])

            #plt.plot(t_values, y_values)
            #plt.show()

            tf = TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstRight, dt=maxdt/100)
            kernels[c][c2] = HawkesKernelTimeFunc(tf)


    baseline = [lamb0_classes for _ in range(nbClasses)]


    hawkes = SimuHawkes(force_simulation=True, baseline=baseline, max_jumps=num_obs, verbose=False)
    hawkes.threshold_negative_intensity()

    for c in range(nbClasses):
        for c2 in range(nbClasses):
            #if c!=c2: continue  # If uncommented: univariate Hawkes process
            hawkes.set_kernel(c, c2, kernels[c][c2])

    hawkes.simulate()

    events = []
    for c, _ in enumerate(baseline):
        events.append([c,0])
    for c in range(len(hawkes.timestamps)):
        for t in hawkes.timestamps[c]:
            events.append([c, t])
    events = np.array(events)

    return events, hawkes

def simulTxt(events, voc_per_class, nbClasses, overlap_voc, words_per_obs, theta0):
    # Generate text
    # Perfectly separated text content
    theta0 = np.array([theta0]*voc_per_class)

    tries = 0
    while True:
        tries += 1
        if tries>10:
            print(f"Textual overlap of {overlap_voc} too hard to compute")
            return [], -1
        try:
            probs = np.array([sorted(np.random.multinomial(1e9, pvals=np.random.dirichlet(theta0)).squeeze(), reverse=True) for c in range(nbClasses)])
            probs = probs/np.sum(probs, axis=-1)[:, None]

            voc_clusters = [np.array(list(range(int(voc_per_class))), dtype=int) + c*voc_per_class for c in range(nbClasses)]

            probs_final = probs
            if overlap_voc>=1-0.001:
                probs_temp = np.zeros((nbClasses, voc_per_class*nbClasses))
                for c in range(nbClasses):
                    probs_temp[c][voc_clusters[c]] = probs[c]

                probs_final = []
                for c in range(nbClasses):
                    probs_final.append(probs_temp[0][voc_clusters[0]])

            elif overlap_voc is not None:
                overlap=-1
                probs_temp = None
                while overlap<overlap_voc:
                    # Overlap
                    for c in range(nbClasses):
                        #voc_clusters[c] -= int(c*voc_per_class*overlap_voc)
                        voc_clusters[c] -= int(c)

                    probs_temp = np.zeros((nbClasses, voc_per_class*nbClasses))
                    for c in range(nbClasses):
                        probs_temp[c][voc_clusters[c]] = probs[c]
                    overlap = compute_overlap(list(range(voc_per_class*nbClasses)), probs_temp)

                probs_final = []
                for c in range(nbClasses):
                    probs_final.append(probs_temp[c][voc_clusters[c]])

                # x = np.array(list(range(voc_per_class*nbClasses)))
                # for c in range(nbClasses):
                #     plt.plot(x, probs_temp[c])
                # plt.show()
            break
        except:
            print(f"Textual overlap {overlap_voc} hard to reach - Try {tries}")

    for c in range(len(voc_clusters)):
        for w in range(len(voc_clusters[c])):
            voc_clusters[c][w] = int(voc_clusters[c][w])

    # Associate a fraction of vocabulary to each observation
    arrtxt = []
    for e in events:
        c_text = int(e[1])
        arrtxt.append(np.random.choice(voc_clusters[c_text], size=words_per_obs, p=probs_final[c_text]))

    return arrtxt, 0

def compute_overlap(x, ys):
    areas = [np.trapz(y, x=x) for y in ys]
    ys = np.array(ys)
    ysecondmin = np.copy(ys)
    ysecondmin[ysecondmin==np.max(ysecondmin, axis=0)] = 0.
    ysecondmin = np.max(ysecondmin, axis=0)
    areaInter = np.trapz(ysecondmin, x=x)

    #len(areas)*
    overlap = 2*areaInter/np.sum(areas)

    return overlap

def save(folder, name, events, arrtxt, lamb0_poisson, lamb0_classes, means, sigs, alpha):
    ensureFolder(folder)
    events = np.insert(events, 3, np.array(list(range(len(events)))), axis=1)  # Index to identify textual content
    events = np.array(list(sorted(events, key= lambda x: x[2])))  # Sort by time
    with open(folder+name+"_events.txt", "w+") as f:
        for i, e in enumerate(events):
            content = ",".join(map(str, list(arrtxt[int(e[3])])))
            txt = str(e[2])+"\t"+content+"\t"+str(e[0])+"\t"+str(e[1])+"\t"+"\n"
            f.write(txt)

    with open(folder+name+"_lamb0.txt", "w+") as f:
        f.write(str(lamb0_poisson)+"\t"+str(lamb0_classes))
    np.savetxt(folder+name+"_means.txt", means)
    np.savetxt(folder+name+"_sigs.txt", sigs)
    np.save(folder+name+"_alpha", alpha)

def RBF_kernel(reference_time, time_interval, bandwidth):
    ''' RBF kernel for Hawkes process.
        @param:
            1.reference_time: np.array, entries larger than 0.
            2.time_interval: float/np.array, entry must be the same.
            3. bandwidth: np.array, entries larger than 0.
        @rtype: np.array
    '''
    numerator = - (time_interval[:, None] - reference_time[None, :]) ** 2 / (2 * bandwidth[None, :] ** 2)
    denominator = (2 * np.pi * bandwidth[None, :] ** 2 ) ** 0.5
    return np.exp(numerator) / denominator

def plotProcess(events, means, sigs, alpha, lamb0):
    colors = ["r", "b", "y", "g", "orange", "cyan","purple"]*10

    maxdt = max(means)+3*max(sigs)
    nbClasses = len(alpha)

    res = 1000
    ranget = np.linspace(0, np.max(events[:, 1]), res)
    tabvals = [[] for _ in range(nbClasses)]
    for t in ranget:
        events_cons = events[events[:, 1]>t-maxdt]
        events_cons = events_cons[events_cons[:, 1] < t]


        clusters_timestamps = events_cons[:, 0]
        timestamps = events_cons[:, 1]


        RBF = RBF_kernel(means, t-timestamps, sigs)  # num_time_points, size_kernel
        unweighted_triggering_kernel = np.zeros((len(alpha), len(means)))
        active_clus_to_ind = {int(c): i for i,c in enumerate(sorted(list(set(clusters_timestamps))))}
        for (clus, trig) in zip(clusters_timestamps, RBF):
            indclus = active_clus_to_ind[int(clus)]
            unweighted_triggering_kernel[indclus] = unweighted_triggering_kernel[indclus] + trig


        for c in range(nbClasses):
            val = np.sum(alpha[c]*unweighted_triggering_kernel)

            # eventsprec = ev[ev[:, 0] == c]  # 0 bc has to be temporal clusters
            # val = kernel(t - eventsprec[:, -1], means, sigs, alpha[c,c]).sum()
            tabvals[c].append(val)
    tabvals = np.array(tabvals)

    for c in range(nbClasses):
        plt.plot(ranget, lamb0+tabvals[c], "-", c=colors[c])

    all_timestamps = events[:, 1]
    all_clus = events[:, 0]
    colors_dots = [colors[int(clus)] for clus in all_clus]
    plt.scatter(all_timestamps, [0]*len(all_timestamps), c=colors_dots, s=1)



def generate(params):
    (folder, DS, nbClasses, num_obs, multivariate,
     overlap_voc, overlap_temp, perc_rand,
     voc_per_class, words_per_obs, theta0,
     lamb0_poisson, lamb0_classes, alpha0, means, sigs) = params

    nbTries = 0
    while True:
        alpha = np.zeros((nbClasses, nbClasses, len(means)))
        for c in range(nbClasses):
            a = np.random.dirichlet([alpha0]*nbClasses*len(means))
            a = a.reshape((nbClasses, len(means)))
            alpha[c]=a
        alpha = np.array(alpha)

        if not multivariate:
            for i in range(len(alpha)):
                for j in range(len(alpha[i])):
                    if i==j: continue
                    else: alpha[i,j]*=0

        alpha = alpha/(alpha.sum(axis=(-1,-2))[:, None, None]+1e-20)

        if overlap_temp is None:
            break

        t = np.linspace(0, np.max(means)+6*np.max(sigs), 1000)
        RBF = RBF_kernel(means, t, sigs)
        kernels = []
        for i in range(len(alpha)):
            for j in range(len(alpha[i])):
                kernels.append(RBF.dot(alpha[i,j]))
        overlap_temp_temp = compute_overlap(t, kernels)
        if (overlap_temp_temp>overlap_temp-0.025 and overlap_temp_temp<overlap_temp+0.025) or nbClasses==1:
            print("Overlap temporel", overlap_temp)
            break
        nbTries += 1
        if nbTries>1000000:
            print(f"Overlap temp = {overlap_temp} too hard to compute")
            return -1

    #print(alpha)



    # Get timestamps and temporal clusters
    events, hawkes = simulHawkes(lamb0_poisson, lamb0_classes, alpha, means, sigs, num_obs=num_obs)
    print(len(events), "events")
    unique, cnt = np.unique(events[:, 0], return_counts=True)
    print(list(cnt))
    visualize = False
    if visualize:
        plotProcess(events, means, sigs, alpha, lamb0_classes)
        plt.show()

        fit = True
        if fit:
            em = HawkesEM(15, kernel_size=30, n_threads=7, verbose=False, tol=1e-3)
            em.fit(hawkes.timestamps)
            fig = plot_hawkes_kernels(em, hawkes=hawkes, show=False)
            plt.show()

        #sys.exit()

    # Initialize textual clusters and shuffle nb_rand of them
    events = np.insert(events, 0, events[:, 0], axis=1)  # Set textual cluster as identical to temporal ones
    nb_rand = int(perc_rand*len(events))
    events[np.random.randint(0, len(events), nb_rand), 1] = np.random.randint(0, nbClasses, nb_rand)  # Shuffle clusters


    try:
        overlaps_voc = overlap_voc.copy()
    except:
        overlaps_voc = [overlap_voc]
    s = int(np.random.random()*1000)

    for overlap_voc in overlaps_voc:
        np.random.seed(s)
        # Generate text associated with textual clusters
        arrtxt, success = simulTxt(events, voc_per_class, nbClasses, overlap_voc, words_per_obs, theta0)

        if success == -1:
            print(f"Textual overlap of {overlap_voc} too hard to compute")
            if len(overlaps_voc)==1:
                return -1

        name = f"Obs_nbclasses={nbClasses}_lg={num_obs}_overlapvoc={overlap_voc}_overlaptemp={overlap_temp}_percrandomizedclus={perc_rand}_vocperclass={voc_per_class}_wordsperevent={words_per_obs}_DS={DS}"
        save(folder, name, events, arrtxt, lamb0_poisson, lamb0_classes, means, sigs, alpha)

    return 0


if __name__ == "__main__":
    nbClasses = 3
    num_obs = 5000

    overlap_voc = 0.5  # Proportion of voc in common between a clusters and its direct neighbours
    overlap_temp = 0.05  # Overlap between the kernels of the simulating process

    voc_per_class = 1000  # Number of words available for each cluster
    perc_rand = 0.  # Percentage of events to which assign random textual cluster
    words_per_obs = 10

    means = np.array([3, 5, 7, 11, 13])
    sigs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])


    lamb0_poisson = 0.05
    lamb0_classes = 0.1
    theta0 = 10
    alpha0 = 0.1

    DS = 0

    multivariate = True

    folder = "data/Synth/"
    np.random.seed(1564)
    params = (folder, DS, nbClasses, num_obs, multivariate,
              overlap_voc, overlap_temp, perc_rand,
              voc_per_class, words_per_obs, theta0,
              lamb0_poisson, lamb0_classes, alpha0, means, sigs)
    generate(params)







