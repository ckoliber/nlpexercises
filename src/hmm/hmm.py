import warnings

import re as re
import numba as nb
import numpy as np
import sklearn.base as skbase

warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=nb.NumbaPendingDeprecationWarning)

@nb.jit(nopython=True)
def _alpha(x, start_probs, transition_probs, observation_probs):
    n_components = start_probs.shape[0]
    alpha = np.zeros((len(x), n_components))
    alpha[0, :] = start_probs[:] * observation_probs[:, x[0]]

    for t in range(1, len(x), 1):
        token = x[t]
        for i in range(n_components):
            alpha[t, i] = (
                alpha[t - 1, :] @ transition_probs[:, i]
            ) * observation_probs[i, token]

    return alpha

@nb.jit(nopython=True)
def _beta(x, start_probs, transition_probs, observation_probs):
    n_components = start_probs.shape[0]
    beta = np.zeros((len(x), n_components))
    beta[-1, :] = np.ones((n_components))

    for t in range(len(x) - 2, -1, -1):
        next_token = x[t + 1]
        for i in range(n_components):
            beta[t, i] = (
                beta[t + 1] * observation_probs[:, next_token]
            ) @ transition_probs[i, :]

    return beta

@nb.jit(nopython=True)
def _xi(x, alpha, beta, start_probs, transition_probs, observation_probs):
    n_components = start_probs.shape[0]
    xi = np.zeros((len(x) - 1, n_components, n_components))

    for t in range(0, len(x) - 1, 1):
        next_token = x[t + 1]
        denominator = (
            (alpha[t, :].T @ transition_probs) *
            observation_probs[:, next_token].T
        ) @ beta[t + 1, :]
        for i in range(n_components):
            numerator = alpha[t, i] * transition_probs[i, :] * (
                observation_probs[:, next_token].T * beta[t + 1, :].T
            )
            xi[t, i, :] = numerator / denominator

    return xi

@nb.jit(nopython=True)
def _gamma(x, xi):
    gamma = np.sum(xi, axis=2)
    gamma = np.vstack(
        (gamma, np.sum(xi[len(x) - 2, :, :], axis=0).reshape((1, -1)))
    )

    return gamma

@nb.jit(nopython=True, parallel=True)
def _score(X, lengths, start_probs, transition_probs, observation_probs):
    R = len(lengths)
    result = np.zeros((R))

    for r in nb.prange(R):
        x = X[lengths[0:r].sum():lengths[0:r+1].sum()]

        alpha = _alpha(x, start_probs, transition_probs, observation_probs)
        result[r] = np.sum(alpha[-1, :])

    return result

@nb.jit(nopython=True, parallel=True)
def _fit(X, lengths, n_iter, n_components, start_probs, transition_probs, observation_probs):
    for _ in range(n_iter):
        R = len(lengths)
        xis = np.zeros((R, lengths[0] - 1, n_components, n_components))
        gammas = np.zeros((R, lengths[0], n_components))

        for r in nb.prange(R):
            x = X[lengths[0:r].sum():lengths[0:r+1].sum()]

            alpha = _alpha(x, start_probs, transition_probs, observation_probs)
            beta = _beta(x, start_probs, transition_probs, observation_probs)
            xis[r] = _xi(x, alpha, beta, start_probs, transition_probs, observation_probs)
            gammas[r] = _gamma(x, xis[r])

        start_probs = np.sum(gammas[:, 0, :], axis=0) / R

        transition_probs = xis.sum(0).sum(0) / gammas[:, 0:-1].sum(0).sum(0).reshape(-1, 1)

        denominator = gammas.sum(0).sum(0).reshape(-1, 1)
        for k in range(observation_probs.shape[1]):
            sum_probs = np.zeros((R, n_components))
            for r in nb.prange(R):
                x = X[lengths[0:r].sum():lengths[0:r+1].sum()]

                sum_probs[r] = gammas[r][x == k, :].sum(0)
            
            observation_probs[:, k] = sum_probs.sum(0)
        
        observation_probs = observation_probs / denominator

    return (start_probs, transition_probs, observation_probs)

class HMMEstimator(skbase.BaseEstimator):
    def __init__(
        self,
        end_prob=1.0,
        start_probs=None,
        transition_probs=None,
        observation_probs=None,
        n_components=1,
        n_iter=10,
    ):
        self.end_prob = end_prob
        self.start_probs = start_probs
        self.transition_probs = transition_probs
        self.observation_probs = observation_probs
        self.n_components = n_components
        self.n_iter = n_iter

    def score(self, X, lengths):
        return _score(
            X,
            lengths,
            self.start_probs,
            self.transition_probs,
            self.observation_probs
        )

    def predict(self, X, lengths):
        pass

    def fit(self, X, lengths):
        (self.start_probs, self.transition_probs, self.observation_probs) = _fit(
            X,
            lengths,
            self.n_iter,
            self.n_components,
            self.start_probs,
            self.transition_probs,
            self.observation_probs
        )

        return self

def load_model(path):
    sections = open(path).read().split("\n\n")

    start_probs = np.array(re.sub(r"^\w+: \d+\n", "", sections[0]).split("\t")).astype(
        float
    )
    transition_probs = np.array(
        [row.split("\t") for row in re.sub(
            r"^\w+: \d+\n", "", sections[1]).split("\n")]
    ).astype(float)
    observation_probs = np.array(
        [
            row.split("\t")
            for row in re.sub(r"^\w+: \d+\n", "", sections[2]).split("\n")[:-1]
        ]
    ).astype(float)

    return HMMEstimator(
        start_probs=start_probs,
        transition_probs=transition_probs,
        observation_probs=observation_probs.T,
        n_components=start_probs.size,
        n_iter=5,
    )

def save_model(path, model):
    text = ""
    text += f"initial: {model.start_probs.shape[0]}\n"
    text += (
        np.array2string(model.start_probs, separator="\t", max_line_width=200)
        .replace("[", "")
        .replace("]", "")
    )
    text += "\n\n"
    text += f"transition: {model.transition_probs.shape[0]}\n"
    text += (
        np.array2string(model.transition_probs,
                        separator="\t", max_line_width=200)
        .replace(" [", "")
        .replace("[", "")
        .replace("]", "")
    )
    text += "\n\n"
    text += f"observation: {model.observation_probs.T.shape[1]}\n"
    text += (
        np.array2string(model.observation_probs.T,
                        separator="\t", max_line_width=200)
        .replace(" [", "")
        .replace("[", "")
        .replace("]", "")
    )
    text += "\n"

    open(path, "w+").write(text)
