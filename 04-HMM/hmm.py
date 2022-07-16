# Author: Kaituo Xu, Fan Yu
import numpy as np


def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    alpha = np.zeros([T, len(pi)])

    alpha[0, :] = pi * B[:, O[0]]
    for t in range(1, T):
        alpha[t, :] = np.sum(alpha[t - 1, :] * A.T, axis=1) * B[:, O[t]]
    return np.sum(alpha[T - 1, :])


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    beta = np.zeros([T, len(pi)])

    beta[T - 1, :] = 1
    for t in reversed(range(T - 1)):
        beta[t, :] = np.sum(A * B[:, O[t + 1]].T * beta[t + 1, :], axis=1)
    return np.sum(pi * B[:, O[0]] * beta[0, :])


def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)

    delta = np.ones([T, len(pi)])
    psi = np.zeros(delta.shape, dtype=int)
    delta[0, :] = pi * B[:, O[0]]
    for t in range(1, T):
        delta[t, :] = np.max(delta[t - 1, :] * A.T * B[:, O[t]], axis=1)
        psi[t, :] = np.argmax(delta[t - 1, :] * A.T, axis=1) + 1

    i = np.zeros(T, dtype=int)
    i[T - 1] = np.argmax(delta[T - 1, :]) + 1
    for t in reversed(range(T - 1)):
        i[t] = psi[t + 1, i[t + 1] - 1]

    return np.max(delta[T - 1, :]), i


if __name__ == "__main__":
    color2id = {"RED": 0, "WHITE": 1}
    # model parameters
    pi = np.array([0.2, 0.4, 0.4])
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)
