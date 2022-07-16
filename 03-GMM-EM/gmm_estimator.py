# Author: Sining Sun , Zhanheng Yang

import numpy as np
from utils import *
import scipy.cluster.vq as vq

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


class GMM:

    def __init__(self, X, D, K=5):
        """ Initialize the parameters of the Gaussian mixture model
            :param X: a matrix including data samples, N * D
            :param D: the dimension of data points, 39 dimensional MFCC features
            :param K: the number of the components in the Gaussian mixture model
        """
        assert (D > 0)
        self.D = D
        self.K = K
        self.pi = []
        self.mu = []
        self.sigma = []
        # K-means initial
        self.kmeans_initial(X)

    def kmeans_initial(self, X):
        """ Initial the Gaussian mixture model with K-means algorithm
            :param X: a matrix including data samples, N * D
        """
        (_, labels) = vq.kmeans2(X, self.K, minit='points', iter=100)
        clusters = [[] for i in range(self.K)]
        for (label, x) in zip(labels, X):
            clusters[label].append(x)

        for cluster in clusters:
            self.pi.append(len(cluster) * 1.0 / len(X))
            self.mu.append(np.mean(cluster, axis=0))
            self.sigma.append(np.cov(cluster, rowvar=0))

    def gaussian(self, x, k):
        """ Calculate gaussion probability of the kth gaussion model
            :param x: the observed data, dim * 1
            :param k: the kth gaussion model
            :return: the gaussion probability, scalor
        """
        det_sigma = np.linalg.det(self.sigma[k])
        inv_sigma = np.linalg.inv(self.sigma[k] + np.finfo(float).eps)
        mahalanobis = np.dot(np.dot((x - self.mu[k]).T, inv_sigma),
                             (x - self.mu[k]))
        const = 1 / ((2 * np.pi)**(self.D / 2))
        return const * det_sigma**(-0.5) * np.exp(-0.5 * mahalanobis)

    def calc_log_likelihood(self, X):
        """ Calculate log likelihood of GMM
            param: X: a matrix including data samples samples, N * D
            return: log likelihood of current model
        """
        N = len(X)
        gamma = np.zeros((self.K, N))
        for k in range(self.K):
            for n in range(N):
                gamma[k, n] = self.pi[k] * self.gaussian(X[n, :], k)
        log_llh = np.sum(np.log(np.sum(gamma, axis=0)))

        return log_llh

    def em_estimator(self, X):
        """ Update paramters of GMM
            param: X: a matrix including data samples samples, N * D
            return: log likelihood of updated model
        """
        # E step
        N = len(X)
        gamma = np.zeros((self.K, N))
        for k in range(self.K):
            for n in range(N):
                gamma[k, n] = self.pi[k] * self.gaussian(X[n, :], k)
        gamma /= np.sum(gamma, axis=0)

        # M step
        self.pi = []
        self.mu = []
        self.sigma = []
        for k in range(self.K):
            Nk = np.sum(gamma[k, :])
            self.pi.append(Nk / N)
            self.mu.append(np.dot(gamma[k, :], X) / Nk)
            self.sigma.append(
                np.dot(gamma[k, :] * (X - self.mu[k]).T, X - self.mu[k]) / Nk)
        log_llh = self.calc_log_likelihood(X)
        return log_llh


def train(gmm, feats, num_iterations=num_iterations):
    for i in range(num_iterations):
        log_llh = gmm.em_estimator(feats)
        print('Iteration {}: {}'.format(i, log_llh))
    return gmm


def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets(
        'test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    dict_utt2feat, dict_target2utt = read_feats_and_targets(
        'train/feats.scp', 'train/text')
    for target in targets:
        print('Initialize and train for target {}'.format(target))
        feats = get_feats(target, dict_utt2feat, dict_target2utt)
        gmms[target] = GMM(feats, D=39, K=num_gaussian)
        gmms[target] = train(gmms[target], feats)
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


if __name__ == '__main__':
    main()