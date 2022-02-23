import numpy as np
from scipy.special import erfc, gammaln, gamma
from copy import deepcopy as copy
from scipy.stats import dirichlet as dir

ones = {i: np.ones((i)) for i in range(1, 100)}

class Document(object):
	def __init__(self, index, timestamp, word_distribution, word_count):
		super(Document, self).__init__()
		self.index = index
		self.timestamp = timestamp
		self.word_distribution = np.array(word_distribution, dtype=int)
		self.word_count = np.array(word_count, dtype=int)
		
class Cluster(object):
	def __init__(self, index, num_samples, active_clusters, alpha0, size_kernel):# alpha, word_distribution, documents, word_count):
		super(Cluster, self).__init__()
		self.index = index
		self.alpha = None
		self.alpha_final = {}
		self.word_distribution = None
		self.word_count = 0
		self.likelihood_samples = np.zeros((num_samples), dtype=np.float)
		self.likelihood_samples_sansLambda = np.zeros((num_samples), dtype=np.float)
		self.triggers = np.zeros((num_samples), dtype=np.float)
		self.integ_triggers = np.zeros((num_samples), dtype=np.float)  # Ones pcq premiere obs incluse

		vectors, priors = draw_vectors(alpha0, num_samples, active_clusters, size_kernel, return_priors=True)
		self.alphas = vectors
		self.log_priors = priors

	def add_document(self, doc):
		if self.word_distribution is None:
			self.word_distribution = np.copy(doc.word_distribution)
		else:
			self.word_distribution += doc.word_distribution
		self.word_count += doc.word_count

	def __repr__(self):
		return 'cluster index:' + str(self.index) + '\n' +'word_count: ' + str(self.word_count) \
		+ '\nalpha:' + str(self.alpha)+"\n"

class Particle(object):
	"""docstring for Particle"""
	def __init__(self, weight):
		super(Particle, self).__init__()
		self.weight = weight
		self.log_update_prob = 0
		self.clusters = {}  # can be store in the process for efficient memory implementation, key = cluster_index, value = cluster object
		self.docs2cluster_ID = []  # the element is the cluster index of a sequence of document ordered by the index of document
		self.all_timestamps = []  # the element is the cluster index of a sequence of document ordered by the index of document
		self.active_clusters = {}  # dict key = cluster_index, value = list of timestamps in specific cluster (queue)
		self.active_timestamps = None
		self.cluster_num_by_now = 0
		self.active_clus_to_ind = None

	def __repr__(self):
		return 'particle document list to cluster IDs: ' + str(self.docs2cluster_ID) + '\n' + 'weight: ' + str(self.weight)
		

def draw_vectors(alpha0, num_samples, active_clusters, size_kernel, method="beta", return_priors=False):
	vec = None
	prior = None
	if method=="dirichlet":
		vec = dirichlet(alpha0, num_samples, active_clusters, size_kernel)
	if method=="beta":
		vec = beta(alpha0, num_samples, active_clusters, size_kernel)
	vec[vec==0.] = 1e-10
	vec[vec==1.] = 1-1e-10

	if not return_priors:
		return vec

	else:
		if method=="dirichlet":
			prior = log_dirichlet_PDF(vec, alpha0)
		if method=="beta":
			prior = log_beta_prior(vec, alpha0)
		return vec, prior

def dirichlet(prior, num_samples, active_clusters, size_kernel):
	''' Draw 1-D samples from a dirichlet distribution
	'''
	draws = np.random.dirichlet([prior]*(size_kernel), size=(num_samples, len(active_clusters)))
	if num_samples==1:
		draws=draws[0]
	return draws

def beta(alpha0, num_samples, active_clusters, size_kernel):
	''' Draw 1-D samples from a dirichlet distribution
	'''
	skew = 1
	draws = np.random.beta(alpha0, skew*alpha0, size=(num_samples, len(active_clusters), size_kernel))
	if num_samples==1:
		draws=draws[0]
	return draws

def multinomial(exp_num, probabilities):
	''' Draw samples from a multinomial distribution.
		@param:
			1. exp_num: Number of experiments.
			2. probabilities: multinomial probability distribution (sequence of floats).
		@rtype: 1-D numpy array
	'''
	return np.random.multinomial(exp_num, probabilities).squeeze()

def EfficientImplementation(tn, reference_time, bandwidth, epsilon = 1e-5):
	''' return the time we need to compute to update the triggering kernel
		@param:
			1.tn: float, current document time
			2.reference_time: list, reference_time for triggering_kernel
			3.bandwidth: int, bandwidth for triggering_kernel
			4.epsilon: float, error tolerance
		@rtype: float
	'''
	max_ref_time = max(reference_time)
	max_bandwidth = max(bandwidth)
	tu = tn - ( max_ref_time + np.sqrt( -2 * max_bandwidth * np.log(0.5 * epsilon * np.sqrt(2 * np.pi * max_bandwidth**2)) ))
	return tu

def log_dirichlet_PDF(alpha, alpha0):
	''' return the logpdf for each entry of a list of dirichlet draws
	'''

	lg = alpha.shape[-1]
	if lg not in ones: ones[lg] = np.ones((lg))
	priors = (np.log(alpha)*(alpha0-1)).dot(ones[lg]) + gammaln(alpha0*lg) - gammaln(alpha0)*lg

	lg = priors.shape[-1]
	if lg not in ones: ones[lg] = np.ones((lg))
	priors = priors.dot(ones[lg])

	return priors

def log_beta_prior(alpha, alpha0):
	''' return the logpdf for each entry of a list of dirichlet draws
	'''

	skew = 1
	priors = np.log(alpha)*(alpha0-1) + np.log(1-alpha)*(skew*alpha0-1) + gammaln(alpha0+skew*alpha0) - gammaln(alpha0) - gammaln(skew*alpha0)

	lg = priors.shape[-1]
	if lg not in ones: ones[lg] = np.ones((lg))
	priors = priors.dot(ones[lg])

	lg = priors.shape[-1]
	if lg not in ones: ones[lg] = np.ones((lg))
	priors = priors.dot(ones[lg])

	return priors

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

def triggering_kernel(alpha, reference_time, time_intervals, bandwidth):
	''' triggering kernel for Hawkes porcess.
		@param:
			1. alpha: np.array, entres larger than 0
			2. reference_time: np.array, entries larger than 0.
			3. time_intervals: float/np.array, entry must be the same.
			4. bandwidth: np.array, entries larger than 0.
		@rtype: np.array
	'''
	#if len(alpha) != len(reference_time):
		#raise Exception("length of alpha and length of reference time must equal")
	#time_intervals = time_intervals.reshape(-1, 1)

	if len(alpha.shape) == 3:
		RBF = RBF_kernel(reference_time, time_intervals, bandwidth)  # num_time_points, size_kernel

		lg = RBF.shape[0]
		if lg not in ones: ones[lg] = np.ones((lg))
		rbfdotted = alpha.dot(ones[lg].dot(RBF))

		return rbfdotted[0]

	else:
		RBF = RBF_kernel(reference_time, time_intervals, bandwidth)  # num_time_points, size_kernel

		rbfdotted = RBF.dot(alpha.T)  # num_time_points, num_active_clusters
		lg = len(rbfdotted)
		if lg==1:
			rbfdotted = rbfdotted[0]
		else:
			if lg not in ones: ones[lg] = np.ones((lg))
			rbfdotted = ones[lg].dot(rbfdotted)

		lg = len(rbfdotted)
		if lg==1:
			return rbfdotted[0]
		if lg not in ones: ones[lg] = np.ones((lg))

		rbfdotted = ones[lg].dot(rbfdotted)

		return rbfdotted

def g_theta(timeseq, reference_time, bandwidth, max_time):
	''' g_theta for DHP
		@param:
			2. timeseq: 1-D np array time sequence before current time
			3. base_intensity: float
			4. reference_time: 1-D np.array
			5. bandwidth: 1-D np.array
		@rtype: np.array, shape(3,)
	'''
	timeseq = np.array(timeseq)
	results = 0.5 * ( erfc( (- reference_time[None, :]) / (2 * bandwidth[None, :] ** 2) ** 0.5) - erfc( (max_time - timeseq[:, None] - reference_time[None, :]) / (2 * bandwidth[None, :] ** 2) **0.5) )

	return results

def update_cluster_likelihoods(active_timestamps, cluster, reference_time, bandwidth, base_intensity, max_time):
	timeseq = active_timestamps[:, 1]
	clusseq = active_timestamps[:, 0]
	num_active_clus = len(set(clusseq))

	alphas = cluster.alphas
	Lambda_0 = base_intensity * max_time
	integ_RBF = g_theta(np.array([timeseq[-1]]), reference_time, bandwidth, max_time)
	unweighted_integ_triggering_kernel = np.zeros((num_active_clus, len(reference_time)))
	indToClus = {int(c): i for i,c in enumerate(sorted(list(set(clusseq))))}
	for (clus, integ_trig) in zip(clusseq, integ_RBF):
		indclus = indToClus[int(clus)]
		unweighted_integ_triggering_kernel[indclus] = unweighted_integ_triggering_kernel[indclus] + integ_trig

	alphas_times_gtheta = np.sum(alphas * unweighted_integ_triggering_kernel[None, :, :], axis=(-1, -2))  # Egal au nombre de points temporel partout je crois, simplifier ?


	time_intervals = timeseq[-1] - timeseq[timeseq<timeseq[-1]]
	if len(time_intervals)!=0:
		RBF = RBF_kernel(reference_time, time_intervals, bandwidth)  # num_time_points, size_kernel
		unweighted_triggering_kernel = np.zeros((num_active_clus, len(reference_time)))
		indToClus = {int(c): i for i,c in enumerate(sorted(list(set(clusseq))))}
		for (clus, trig) in zip(clusseq, RBF):
			indclus = indToClus[int(clus)]
			unweighted_triggering_kernel[indclus] = unweighted_triggering_kernel[indclus] + trig


		cluster.triggers = np.sum(alphas*unweighted_triggering_kernel[None, :, :], axis=(-1, -2))

		cluster.likelihood_samples_sansLambda += np.log(cluster.triggers)

	cluster.integ_triggers += alphas_times_gtheta

	cluster.likelihood_samples = -Lambda_0 - cluster.integ_triggers + cluster.likelihood_samples_sansLambda

	return cluster

def update_triggering_kernel_optim(cluster, alpha_true=None, particle = None, DirProc=None):
	''' procedure of triggering kernel for SMC
		@param:
			1. timeseq: list, time sequence including current time
			2. alphas: 2-D np.array with shape (sample number, length of alpha)
			3. reference_time: np.array
			4. bandwidth: np.array
			5. log_priors: 1-D np.array with shape (sample number,), p(alpha, alpha_0)
			6. base_intensity: float
			7. max_time: float
		@rtype: 1-D numpy array with shape (length of alpha0,)
	'''
	alphas = cluster.alphas
	log_priors = cluster.log_priors
	logLikelihood = cluster.likelihood_samples
	log_update_weight = log_priors + logLikelihood
	log_update_weight = log_update_weight - np.max(log_update_weight)  # Prevents overflow
	update_weight = np.exp(log_update_weight)

	#update_weight[update_weight<np.mean(update_weight)]=0.  # Removes noise of obviously unfit alpha samples

	lg = len(update_weight)
	if lg not in ones: ones[lg] = np.ones((lg))
	sumUpdateWeight = update_weight.dot(ones[lg])
	update_weight = update_weight / sumUpdateWeight

	#update_weight = update_weight.reshape(-1,1)
	#alpha = np.sum(update_weight * alphas, axis = 0)
	alpha = np.tensordot(update_weight, alphas, axes=1)

	if len(particle.active_clusters)==2 and False:
		#alpha_true = alphas[0]
		events = list(zip(particle.docs2cluster_ID, particle.all_timestamps))
		events = np.array(events)
		loglik = 0
		T=0
		#print(particle.docs2cluster_ID)
		triggerstot= 0.
		for c,t in events:
			c = c-1
			c = int(c)
			T = t + DirProc.horizon
			tu = EfficientImplementation(t, DirProc.reference_time, DirProc.bandwidth, epsilon=1e-10)
			events_cons = events[events[:, 1]>tu]
			events_cons = events_cons[events_cons[:, 1] <= t]
			#events_cons = np.array([e for e in events_cons if e[0] in particle.active_clusters])
			if len(events_cons)==0: continue

			clusters_timestamps = events_cons[:, 0]
			timestamps = events_cons[:, 1]
			time_intervals = t - timestamps[timestamps<t]

			integ_RBF = g_theta(np.array([t]), DirProc.reference_time, DirProc.bandwidth, T)
			unweighted_integ_triggering_kernel = np.zeros((len(alpha), len(DirProc.reference_time)))
			indToClus = {int(ci): i for i,ci in enumerate(sorted(list(set(clusters_timestamps))))}
			for (clus, integ_trig) in zip(clusters_timestamps, integ_RBF):
				indclus = indToClus[int(clus)]
				unweighted_integ_triggering_kernel[indclus] = unweighted_integ_triggering_kernel[indclus] + integ_trig

			integ_trigger = np.sum(alpha_true[c] * unweighted_integ_triggering_kernel[None, :, :], axis=(-1, -2))  # Egal au nombre de points temporel partout je crois, simplifier ?

			loglik -= integ_trigger

			triggers = 0.
			if len(time_intervals)!=0:
				RBF = RBF_kernel(DirProc.reference_time, time_intervals, DirProc.bandwidth)  # num_time_points, size_kernel
				unweighted_triggering_kernel = np.zeros((len(alpha), len(DirProc.reference_time)))
				active_clus_to_ind = {int(ci): i for i,ci in enumerate(sorted(list(set(clusters_timestamps))))}
				for (clus, trig) in zip(clusters_timestamps, RBF):
					indclus = active_clus_to_ind[int(clus)]
					unweighted_triggering_kernel[indclus] = unweighted_triggering_kernel[indclus] + trig

				triggers = np.sum(alpha_true[c]*unweighted_triggering_kernel[None, :, :], axis=(-1, -2))

				triggerstot += np.log(triggers)
				loglik += np.log(triggers)



		loglik -= DirProc.base_intensity * T

		# print(loglik, np.max(logLikelihood))
	# print(np.max(logLikelihood), np.min(logLikelihood), np.mean(logLikelihood))
	# print(np.max(update_weight), np.min(update_weight), np.mean(update_weight))#, update_weight)
	# print(alpha_true)
	# print(alpha)
	# print(alphas[np.where(logLikelihood==np.max(logLikelihood))[0]])#, alpha)
	# print()


	return alpha

def log_dirichlet_multinomial_distribution(cls_word_distribution, doc_word_distribution, cls_word_count, doc_word_count, vocabulary_size, priors):
	''' compute the log dirichlet multinomial distribution
		@param:
			1. cls_word_distribution: 1-D numpy array, including document word_distribution
			2. doc_word_distribution: 1-D numpy array
			3. cls_word_count: int, including document word_distribution
			4. doc_word_count: int
			5. vocabulary_size: int
			6. priors: 1-d np.array
		@rtype: float
	'''

	#arrones = np.ones((len(priors)))
	#priors_sum = np.sum(priors)
	#priors_sum = priors.dot(arrones)
	priors_sum = priors[0]*len(priors)  # ATTENTION PRIOR[0] SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!
	log_prob = 0
	log_prob += gammaln(cls_word_count - doc_word_count + priors_sum)
	log_prob -= gammaln(cls_word_count + priors_sum)

	#log_prob += np.sum(gammaln(cls_word_distribution + priors))
	#log_prob -= np.sum(gammaln(cls_word_distribution - doc_word_distribution + priors))

	#log_prob += gammaln(cls_word_distribution + priors).dot(arrones)
	#log_prob -= gammaln(cls_word_distribution - doc_word_distribution + priors).dot(arrones)

	cnt = np.bincount(cls_word_distribution)
	un = cnt.nonzero()[0]
	cnt = cnt[un]

	log_prob += gammaln(un + priors[0]).dot(cnt)  # ATTENTION SI PRIOR[0] SEULEMENT SI THETA0 EST SYMMETRIQUE !!!!

	cnt = np.bincount(cls_word_distribution-doc_word_distribution)
	un = cnt.nonzero()[0]
	cnt = cnt[un]

	log_prob -= gammaln(un + priors[0]).dot(cnt)

	return log_prob
