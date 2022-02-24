import pickle
import sys
import os
import numpy as np
import time
from copy import deepcopy as copy
from utils import *
from sklearn.metrics import normalized_mutual_info_score as NMI
import gzip

np.random.seed(12345)

"""
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")
pause()
"""

class Dirichlet_Hawkes_Process(object):
	"""docstring for Dirichlet Hawkes Prcess"""
	def __init__(self, particle_num, base_intensity, theta0, alpha0, reference_time, vocabulary_size, bandwidth, sample_num, r, multivariate):
		super(Dirichlet_Hawkes_Process, self).__init__()
		self.r = r
		self.multivariate = multivariate
		self.particle_num = particle_num
		self.base_intensity = base_intensity
		self.theta0 = theta0
		self.alpha0 = alpha0
		self.reference_time = reference_time
		self.vocabulary_size = vocabulary_size
		self.bandwidth = bandwidth
		self.horizon = (max(self.reference_time)+max(self.bandwidth))
		self.sample_num = sample_num
		self.particles = []
		for i in range(particle_num):
			self.particles.append(Particle(weight = 1.0 / self.particle_num))

		self.active_interval = None

	def sequential_monte_carlo(self, doc, threshold):
		# Set relevant time interval
		tu = EfficientImplementation(doc.timestamp, self.reference_time, self.bandwidth, epsilon=1e-10)
		T = doc.timestamp + self.horizon  # So that Gaussian RBF kernel is fully computed; needed to correctly compute the integral part of the likelihood
		self.active_interval = [tu, T]

		particles = []
		for particle in self.particles:
			particles.append(self.particle_sampler(particle, doc))

		self.particles = particles

		# Resample particules whose weight is below the given threshold
		self.particles = self.particles_normal_resampling(self.particles, threshold)

	def particle_sampler(self, particle, doc):
		# Sample cluster label
		particle, selected_cluster_index = self.sampling_cluster_label(particle, doc)
		# Update the triggering kernel
		particle.clusters[selected_cluster_index].alpha = self.parameter_estimation(particle, selected_cluster_index)
		# Calculate the weight update probability ; + bc log form
		particle.log_update_prob = self.calculate_particle_log_update_prob(particle, selected_cluster_index, doc)
		return particle

	def sampling_cluster_label(self, particle, doc):
		if len(particle.clusters) == 0: # The first document is observed
			particle.cluster_num_by_now += 1
			selected_cluster_index = particle.cluster_num_by_now
			particle.active_clusters[selected_cluster_index] = [doc.timestamp]
			particle.active_clus_to_ind = {int(c): i for i,c in enumerate(sorted(particle.active_clusters))}
			particle.active_timestamps = np.array([[selected_cluster_index, doc.timestamp]])
			selected_cluster = Cluster(index = selected_cluster_index, num_samples=self.sample_num,
									   active_clusters=particle.active_clusters, alpha0=self.alpha0,
									   size_kernel=len(self.reference_time),
									   multivariate=self.multivariate, index_cluster=particle.active_clus_to_ind[selected_cluster_index])
			selected_cluster.add_document(doc)
			particle.clusters[selected_cluster_index] = selected_cluster #.append(selected_cluster)
			particle.docs2cluster_ID.append(selected_cluster_index)
			self.active_cluster_logrates = {0:0, 1:0}

			particle.clusters = self.update_clusters_samples(particle)

		else: # A new document arrives
			active_cluster_indexes = [0] # Zero for new cluster
			active_cluster_rates = [self.base_intensity]
			cls0_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(doc.word_distribution, doc.word_distribution,\
			 doc.word_count, doc.word_count, self.vocabulary_size, self.theta0)
			active_cluster_textual_probs = [cls0_log_dirichlet_multinomial_distribution]
			# Update list of relevant timestamps
			particle = self.update_active_clusters(particle)

			# Posterior probability for each cluster
			clusters_timeseq = particle.active_timestamps[particle.active_timestamps[:, 1]<doc.timestamp, 0]
			timeseq = doc.timestamp - particle.active_timestamps[particle.active_timestamps[:, 1]<doc.timestamp, 1]
			particle.active_clus_to_ind = {int(c): i for i,c in enumerate(sorted(particle.active_clusters))}
			if len(timeseq)==0:  # Because kernels are rigorously null at this point, making this the only possible choice
				selected_cluster_index=0
			else:
				RBF = RBF_kernel(self.reference_time, timeseq, self.bandwidth)  # num_time_points, size_kernel
				unweighted_triggering_kernel = np.zeros((len(particle.active_clusters), len(self.reference_time)))
				for clus in set(clusters_timeseq):
					indclus = particle.active_clus_to_ind[int(clus)]
					trigg = RBF[clusters_timeseq==clus]
					nb = len(trigg)
					if nb==1:
						unweighted_triggering_kernel[indclus] = trigg[0]
						continue
					if nb not in ones: ones[nb] = np.ones((nb))
					unweighted_triggering_kernel[indclus] = ones[nb].dot(trigg)

				for active_cluster_index in particle.active_clusters:
					active_cluster_indexes.append(active_cluster_index)
					alpha = particle.clusters[active_cluster_index].alpha

					mat=unweighted_triggering_kernel*alpha
					lg = mat.shape[0]
					if lg not in ones: ones[lg] = np.ones((lg))
					mat = ones[lg].dot(mat)
					lg = mat.shape[0]
					if lg not in ones: ones[lg] = np.ones((lg))
					mat = ones[lg].dot(mat)
					rate = mat

					# Powered Dirichlet-Hawkes prior
					active_cluster_rates.append(rate)

					# Language model likelihood
					cls_word_distribution = particle.clusters[active_cluster_index].word_distribution + doc.word_distribution
					cls_word_count = particle.clusters[active_cluster_index].word_count + doc.word_count
					cls_log_dirichlet_multinomial_distribution = log_dirichlet_multinomial_distribution(cls_word_distribution, doc.word_distribution, cls_word_count, doc.word_count, self.vocabulary_size, self.theta0)
					active_cluster_textual_probs.append(cls_log_dirichlet_multinomial_distribution)

				# Posteriors to probabilities
				active_cluster_logrates = self.r*np.log(np.array(active_cluster_rates)+1e-100)
				self.active_cluster_logrates = {c: active_cluster_logrates[i+1] for i, c in enumerate(particle.active_clusters)}
				self.active_cluster_logrates[0] = active_cluster_logrates[0]
				cluster_selection_probs = active_cluster_logrates + active_cluster_textual_probs # in log scale
				cluster_selection_probs = cluster_selection_probs - max(cluster_selection_probs) # prevent overflow
				cluster_selection_probs = np.exp(cluster_selection_probs)
				cluster_selection_probs = cluster_selection_probs / np.ones((len(cluster_selection_probs))).dot(cluster_selection_probs)  # Normalize

				# print(cluster_selection_probs, active_cluster_indexes)
				# # Random cluster selection
				# selected_cluster_array = multinomial(exp_num = 1, probabilities = cluster_selection_probs)
				# selected_cluster_index = np.array(active_cluster_indexes)[np.nonzero(selected_cluster_array)][0]

				endo_exo_probs = [cluster_selection_probs[0], sum(cluster_selection_probs[1:])]
				endo_exo = np.random.choice([0,1], p=endo_exo_probs)

				# Answers the vanishing prior problem
				selected_cluster_index = None
				if endo_exo==0:
					selected_cluster_index = 0
				elif endo_exo==1:
					lg = len(cluster_selection_probs[1:])
					cluster_selection_probs = cluster_selection_probs[1:]/cluster_selection_probs[1:].dot(ones[lg])
					active_cluster_indexes = active_cluster_indexes[1:]
					selected_cluster_index = np.random.choice(active_cluster_indexes, p=cluster_selection_probs)  # Categorical distribution

				# selected_cluster_index = np.random.choice(active_cluster_indexes, p=cluster_selection_probs)

			# New cluster drawn
			if selected_cluster_index == 0:
				particle.cluster_num_by_now += 1
				selected_cluster_index = particle.cluster_num_by_now
				self.active_cluster_logrates[selected_cluster_index] = self.active_cluster_logrates[0]
				particle.active_clusters[selected_cluster_index] = [doc.timestamp]
				particle.active_clus_to_ind = {int(c): i for i,c in enumerate(sorted(particle.active_clusters))}
				selected_cluster = Cluster(index = selected_cluster_index, num_samples=self.sample_num,
										   active_clusters=particle.active_clusters, alpha0=self.alpha0,
										   size_kernel=len(self.reference_time),
										   multivariate=self.multivariate, index_cluster=particle.active_clus_to_ind[selected_cluster_index])
				selected_cluster.add_document(doc)
				particle.clusters[selected_cluster_index] = selected_cluster
				particle.docs2cluster_ID.append(selected_cluster_index)

				particle.clusters = self.update_clusters_samples(particle)

			# Existing cluster drawn
			else:
				selected_cluster = particle.clusters[selected_cluster_index]
				selected_cluster.add_document(doc)
				particle.docs2cluster_ID.append(selected_cluster_index)
				particle.active_clusters[selected_cluster_index].append(doc.timestamp)

			# Initialized in if==0
			particle.active_timestamps = np.vstack((particle.active_timestamps, [selected_cluster_index, doc.timestamp]))

		particle.all_timestamps.append(doc.timestamp)

		return particle, selected_cluster_index

	def update_clusters_samples(self, particle):
		for cluster_index in particle.active_clusters:
			if particle.clusters[cluster_index].alphas.shape[:2] == (self.sample_num, len(particle.active_clusters)):
				continue
			newVec, newPriors = draw_vectors(self.alpha0, self.sample_num, [0], len(self.reference_time), return_priors=True)
			newVec = newVec.squeeze()
			if not self.multivariate:
				newVec *= 0.
				newVec += 1e-10
				newPriors *= 0.
			particle.clusters[cluster_index].log_priors = particle.clusters[cluster_index].log_priors + newPriors
			particle.clusters[cluster_index].alphas = np.hstack((particle.clusters[cluster_index].alphas, newVec[:, None, :]))
			if particle.clusters[cluster_index].alpha is not None:
				particle.clusters[cluster_index].alpha = np.vstack((particle.clusters[cluster_index].alpha, newVec[0, None, :]))
			else:
				particle.clusters[cluster_index].alpha = newVec[0, None, :]
		return particle.clusters

	def parameter_estimation(self, particle, selected_cluster_index):

		# Observation is alone in the cluster => the cluster is new => random initialization of alpha
		# Note that it cannot be a previously filled cluster since it would have 0 chance to get selected (see sampling_cluster_label)
		# if particle.clusters[selected_cluster_index].alpha is None:
		# 	alpha = draw_vectors(self.alpha0, num_samples=1, active_clusters=particle.active_clusters, size_kernel=len(self.reference_time))
		# 	return alpha

		T = self.active_interval[1]
		particle.clusters[selected_cluster_index] = update_cluster_likelihoods(particle.active_timestamps, particle.clusters[selected_cluster_index], self.reference_time, self.bandwidth, self.base_intensity, T)
		alpha = update_triggering_kernel_optim(particle.clusters[selected_cluster_index])

		for clus in particle.active_clusters:
			particle.clusters[selected_cluster_index].alpha_final[clus] = alpha[particle.active_clus_to_ind[clus]]

		return alpha

	def update_active_clusters(self, particle):
		tu = self.active_interval[0]
		keys = list(particle.active_clusters.keys())
		particle.active_timestamps = particle.active_timestamps[particle.active_timestamps[:, 1]>tu]
		toRem = []
		for cluster_index in keys:
			timeseq = np.array(particle.active_clusters[cluster_index])
			# active_timeseq = [t for t in timeseq if t > tu]
			active_timeseq = timeseq[timeseq>tu]
			if active_timeseq.size==0:
				toRem.append(cluster_index)
			else:
				particle.active_clusters[cluster_index] = list(active_timeseq)

		for cluster_index in sorted(toRem, reverse=True):
			del particle.active_clusters[cluster_index]  # If no observation is relevant anymore, the cluster has 0 chance to get chosen => we remove it from the calculations
			del particle.clusters[cluster_index].alphas
			del particle.clusters[cluster_index].log_priors
			del particle.clusters[cluster_index].likelihood_samples
			del particle.clusters[cluster_index].likelihood_samples_sansLambda
			del particle.clusters[cluster_index].triggers
			del particle.clusters[cluster_index].integ_triggers

			for cluster_index_left in keys:
				if cluster_index_left not in toRem:
					particle.clusters[cluster_index_left].alphas = np.delete(particle.clusters[cluster_index_left].alphas, particle.active_clus_to_ind[cluster_index], axis=1)
					particle.clusters[cluster_index_left].alpha = np.delete(particle.clusters[cluster_index_left].alpha, particle.active_clus_to_ind[cluster_index], axis=0)

		return particle
	
	def calculate_particle_log_update_prob(self, particle, selected_cluster_index, doc):
		cls_word_distribution = particle.clusters[selected_cluster_index].word_distribution
		cls_word_count = particle.clusters[selected_cluster_index].word_count
		doc_word_distribution = doc.word_distribution
		doc_word_count = doc.word_count

		log_update_prob = log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, self.vocabulary_size, self.theta0)

		div = self.active_cluster_logrates.values()
		lg = len(div)
		if lg not in ones: ones[lg] = np.ones((lg))

		lograte = self.active_cluster_logrates[selected_cluster_index]
		lograte = lograte - np.log(ones[lg].dot(np.exp(list(self.active_cluster_logrates.values()))))

		#log_update_prob += lograte

		return log_update_prob

	def particles_normal_resampling(self, particles, threshold):
		weights = []; log_update_probs = []
		for particle in particles:
			weights.append(particle.weight)
			log_update_probs.append(particle.log_update_prob)
		weights = np.array(weights)
		log_update_probs = np.array(log_update_probs)
		log_update_probs = log_update_probs - max(log_update_probs) # Prevents overflow
		update_probs = np.exp(log_update_probs)
		weights = weights * update_probs
		weights = weights / np.sum(weights) # normalization
		# if len(self.particles[0].active_clusters)==1:
		# 	print(np.round(update_probs, 2))
		# 	print(np.round(weights, 4))
		# 	print([len(part.active_clusters) for part in self.particles])
		# 	print()
		resample_num = len(np.where(weights + 1e-5 < threshold)[0])

		if resample_num == 0: # No need to resample particle, but still need to assign the updated weights to particles
			for i, particle in enumerate(particles):
				particle.weight = weights[i]
			return particles
		else:
			remaining_particles = [particle for i, particle in enumerate(particles) if weights[i] + 1e-5 > threshold ]
			resample_probs = weights[np.where(weights + 1e-5 > threshold)]
			resample_probs = resample_probs/np.sum(resample_probs)
			remaining_particle_weights = weights[np.where(weights + 1e-5 > threshold)]
			for i,_ in enumerate(remaining_particles):
				remaining_particles[i].weight = remaining_particle_weights[i]

			resample_distribution = multinomial(exp_num = resample_num, probabilities = resample_probs)
			if not resample_distribution.shape: # The case of only one particle left
				for _ in range(resample_num):
					new_particle = copy(remaining_particles[0])
					remaining_particles.append(new_particle)
			else: # The case of more than one particle left
				for i, resample_times in enumerate(resample_distribution):
					for _ in range(resample_times):
						new_particle = copy(remaining_particles[i])
						remaining_particles.append(new_particle)

			# Normalize the particle weights
			update_weights = np.array([particle.weight for particle in remaining_particles]); update_weights = update_weights / np.sum(update_weights)
			for i, particle in enumerate(remaining_particles):
				particle.weight = update_weights[i]

			self.particles = None
			return remaining_particles

def getArgs(args):
	import re
	dataFile, kernelFile, outputFolder, r, nbRuns, theta0, alpha0, sample_num, particle_num, printRes = [None]*10
	for a in args:
		print(a)
		try: dataFile = re.findall("(?<=data_file=)(.*)(?=)", a)[0]
		except: pass
		try: kernelFile = re.findall("(?<=kernel_file=)(.*)(?=)", a)[0]
		except: pass
		try: outputFolder = re.findall("(?<=output_folder=)(.*)(?=)", a)[0]
		except: pass
		try: r = re.findall("(?<=r=)(.*)(?=)", a)[0]
		except: pass
		try: nbRuns = int(re.findall("(?<=runs=)(.*)(?=)", a)[0])
		except: pass
		try: theta0 = float(re.findall("(?<=theta0=)(.*)(?=)", a)[0])
		except: pass
		try: alpha0 = float(re.findall("(?<=alpha0=)(.*)(?=)", a)[0])
		except: pass
		try: sample_num = int(re.findall("(?<=number_samples=)(.*)(?=)", a)[0])
		except: pass
		try: particle_num = int(re.findall("(?<=number_particles=)(.*)(?=)", a)[0])
		except: pass
		try: printRes = bool(re.findall("(?<=print_progress=)(.*)(?=)", a)[0])
		except: pass

	if dataFile is None:
		sys.exit("Enter a valid value for data_file")
	if kernelFile is None:
		sys.exit("Enter a valid value for kernel_file")
	if outputFolder is None:
		sys.exit("Enter a valid value for output_folder")
	if r is None: print("r value not found; defaulted to 1"); r="1"
	if nbRuns is None: print("nbRuns value not found; defaulted to 1"); nbRuns=1
	if theta0 is None: print("theta0 value not found; defaulted to 0.01"); theta0=0.01
	if alpha0 is None: print("alpha0 value not found; defaulted to 0.5"); alpha0=0.5
	if sample_num is None: print("sample_num value not found; defaulted to 2000"); sample_num=2000
	if particle_num is None: print("particle_num value not found; defaulted to 8"); particle_num=8
	if printRes is None: print("printRes value not found; defaulted to True"); printRes=True

	with open(kernelFile, 'r') as f:
		i=0
		tabMeans, tabSigs = [], []
		for line in f:
			if line=="\n":
				i += 1
				continue
			if i==0:
				lamb0 = float(line.replace("\n", ""))
			if i==1:
				tabMeans.append(float(line.replace("\n", "")))
			if i==2:
				tabSigs.append(float(line.replace("\n", "")))

	ensureFolder(outputFolder)

	if len(tabMeans)!=len(tabSigs):
		sys.exit("The means and standard deviation do not match. Please check the parameters file.\n"
				 "The values should be organized as follows:\n[lambda_0]\n\n[mean_1]\n[mean_2]\n...\n[mean_K]\n\n[sigma_1]\n[sigma_2]\n...\n[sigma_K]\n")
	means = np.array(tabMeans)
	sigs = np.array(tabSigs)

	rarr = []
	for rstr in r.split(","):
		rarr.append(float(rstr))
	return dataFile, outputFolder, means, sigs, lamb0, rarr, nbRuns, theta0, alpha0, sample_num, particle_num, printRes

def ensureFolder(folder):
	curfol = "./"
	for fol in folder.split("/")[:-1]:
		if fol not in os.listdir(curfol) and fol!="":
			os.mkdir(curfol+fol)
		curfol += fol+"/"

def parse_newsitem_2_doc(news_item, vocabulary_size):
	index = news_item[0]
	timestamp = news_item[1]
	word_id = news_item [2][0]
	count = news_item[2][1]
	word_distribution = np.zeros(vocabulary_size)
	word_distribution[word_id] = count
	word_count = np.sum(count)
	doc = Document(index, timestamp, word_distribution, word_count)
	return doc

def readObservations(folder, name, outputFolder):
	dataFile = folder+name+"_events.txt"
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
	with open(outputFolder+name+"_indexWords.txt", "w+", encoding="utf-8") as f:
		for wd in wdToIndex:
			f.write(f"{wdToIndex[wd]}\t{wd}\n")
	V = len(wdToIndex)
	indexToWd = {idx: wd for wd, idx in wdToIndex.items()}
	return observations, V, indexToWd

def saveDHP(DHP, folderOut, nameOut, date=-1):
	while True:
		try:
			writeParticles(DHP, folderOut, nameOut, date)
			break
		except Exception as e:
			print("ERREUR", e)
			time.sleep(10)
			continue

def writeParticles(DHP, folderOut, nameOut, time):
	DHP_copy = copy(DHP)
	for i in range(len(DHP_copy.particles)):
		for c in DHP_copy.particles[i].clusters:
			if c in DHP_copy.particles[i].active_clusters:
				del DHP_copy.particles[i].clusters[c].alphas
				del DHP_copy.particles[i].clusters[c].log_priors
				del DHP_copy.particles[i].clusters[c].alpha
				del DHP_copy.particles[i].clusters[c].likelihood_samples
				del DHP_copy.particles[i].clusters[c].likelihood_samples_sansLambda
				del DHP_copy.particles[i].clusters[c].triggers
				del DHP_copy.particles[i].clusters[c].integ_triggers

	if time==-1: txtTime = "_final"
	else: txtTime = f"_obs={int(time)}"

	with gzip.open(folderOut+nameOut+txtTime+"_particles.pkl.gz", "w+") as f:
		pickle.dump(DHP_copy, f)

def run_fit(observations, folderOut, nameOut, lamb0, means, sigs, r=1., theta0=None, alpha0 = None, sample_num=2000, particle_num=8, printRes=False, vocabulary_size=None, multivariate=True, eval_on_go=False, indexToWd=None):
	"""
	observations = ([array int] index_obs, [array float] timestamp, ([array int] unique_words, [array int] count_words), [opt, int] temporal_cluster, [opt, int] textual_cluster)
	folderOut = Output folder for the results
	nameOut = Name of the file to which _particles_compressed.pklbz2 will be added
	lamb0 = base intensity
	means, sigs = means and sigmas of the gaussian RBF kernel
	r = exponent parameter of the Powered Dirichlet process; defaults to 1. (standard Dirichlet process)
	theta0 = value of the language model symmetric Dirichlet prior
	alpha0 = symmetric Dirichlet prior from which samples used in Gibbs sampling are drawn (estimation of alpha)
	sample_num = number of samples used in Gibbs sampling
	particle_num = number of particles used in the Sequential Monte-Carlo algorithm
	printRes = whether to print the results according to ground-truth (optional parameters of observations and alpha)
	alphaTrue = ground truth alpha matrix used to generate the observations from gaussian RBF kernel
	"""

	ensureFolder(folderOut)

	if vocabulary_size is None:
		allWds = set()
		for a in observations:
			for w in a[2][0]:
				allWds.add(w)
		vocabulary_size = len(list(allWds))+2
	if theta0 is None: theta0 = 1.
	if alpha0 is None: alpha0 = 1.

	particle_num = particle_num
	base_intensity = lamb0
	reference_time = means
	bandwidth = sigs
	theta0 = np.array([theta0 for _ in range(vocabulary_size)])
	sample_num = sample_num
	threshold = 1.0 / (particle_num*2.)

	DHP = Dirichlet_Hawkes_Process(particle_num = particle_num, base_intensity = base_intensity, theta0 = theta0,
								   alpha0 = alpha0, reference_time = reference_time, vocabulary_size = vocabulary_size,
								   bandwidth = bandwidth, sample_num = sample_num, r=r, multivariate=multivariate)

	t = time.time()


	lgObs = len(observations)
	trueClus = []
	for i, news_item in enumerate(observations):
		doc = parse_newsitem_2_doc(news_item = news_item, vocabulary_size = vocabulary_size)
		DHP.sequential_monte_carlo(doc, threshold)


		if (i%100==1 and printRes) or (i>0 and False):
			print(f'r={r} - Handling document {i}/{lgObs} (t={np.round(news_item[1]-observations[0][1], 1)}) - '
				  f'Average time : {np.round((time.time()-t)*1000/(i), 0)}ms - '
				  f'Remaining time : {np.round((time.time()-t)*(len(observations)-i)/(i*3600), 2)}h - '
				  f'ClusTot={DHP.particles[0].cluster_num_by_now} - ActiveClus = {len(DHP.particles[0].active_clusters)}')


		if eval_on_go and printRes:
			trueClus.append(int(float(news_item[-1][0])))
			if (i%100==1 and printRes) or (i>0 and False):
				inferredClus = DHP.particles[0].docs2cluster_ID
				print("NMI", NMI(trueClus, inferredClus), " - NMI_last", NMI(trueClus[-1000:], inferredClus[-1000:]))

			keys = DHP.particles[0].active_clusters.keys()
			for c in keys:
				for c2 in keys:
					if c2 in DHP.particles[0].clusters[c].alpha_final:
						pass
						#print(c, c2, DHP.particles[0].clusters[c].alpha_final[c2])

		if i%5000==1:
			saveDHP(DHP, folderOut, nameOut, date=-1)

	saveDHP(DHP, folderOut, nameOut, date=-1)
	return DHP


if __name__ == '__main__':
	try:
		dataFile, outputFolder, means, sigs, lamb0, arrR, nbRuns, theta0, alpha0, sample_num, particle_num, printRes = getArgs(sys.argv)
	except:
		nbClasses = 2
		num_obs = 500
		XP = "Overlap"

		overlap_voc = 0.5  # Proportion of voc in common between a clusters and its direct neighbours
		overlap_temp = 0.05  # Overlap between the kernels of the simulating process

		voc_per_class = 1000  # Number of words available for each cluster
		perc_rand = 0.  # Percentage of events to which assign random textual cluster
		words_per_obs = 10

		DS = 0

		lamb0 = 0.035  # Cannot be inferred
		theta0 = 10.  # Has already been documented for RW in LDA like models, DHP, etc ~0.01, 0.001
		alpha0 = 1.  # Uniform beta or Dirichlet prior
		means = np.array([3, 5, 7, 11, 13])
		sigs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

		folder = "data/Synth/"
		nameData = f"Obs_nbclasses={nbClasses}_lg={num_obs}_overlapvoc={overlap_voc}_overlaptemp={overlap_temp}" \
				   f"_percrandomizedclus={perc_rand}_vocperclass={voc_per_class}_wordsperevent={words_per_obs}_DS={DS}"
		dataFile = folder+nameData+"_events.txt"
		outputFolder = folder.replace("data/", "output/")
		alpha_true = np.load(folder+nameData+"_alpha.npy")

		arrR = [1.]
		nbRuns = 1
		sample_num = 2000
		particle_num = 8
		multivariate = True
		printRes = True


	observations, V, indexToWd = readObservations(folder, nameData, outputFolder)

	t = time.time()
	i = 0
	nbRunsTot = nbRuns*len(arrR)


	for run in range(nbRuns):
		for r in arrR:
			name = f"{dataFile[dataFile.rfind('/'):].replace('_events.txt', '')}_r={r}" \
				   f"_theta0={theta0}_alpha0={alpha0}_lamb0={lamb0}" \
				   f"_samplenum={sample_num}_particlenum={particle_num}_run={run}"

			# import pprofile
			# profiler = pprofile.Profile()
			# with profiler:

			run_fit(observations, outputFolder, name, lamb0, means, sigs, r=r, theta0=theta0, alpha0=alpha0,
					sample_num=sample_num, particle_num=particle_num, printRes=printRes,
					vocabulary_size=V, multivariate=multivariate, alpha_true=alpha_true, indexToWd=indexToWd)

			# profiler.print_stats()
			# profiler.dump_stats("Benchmark.txt")
			# pause()

			print(f"r={r} - RUN {run}/{nbRuns} COMPLETE - REMAINING TIME: {np.round((time.time()-t)*(nbRunsTot-i)/((i+1e-20)*3600), 2)}h - ELAPSED TIME: {np.round((time.time()-t)/(3600), 2)}h")
			i += 1



