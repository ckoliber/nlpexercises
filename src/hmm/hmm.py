import warnings

import re as re
import numba as nb
import numpy as np
import sklearn.base as skbase

warnings.filterwarnings("ignore", category=nb.NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=nb.NumbaPendingDeprecationWarning)

@nb.jit(nopython=True)
def _delta(x, start_probs, transition_probs, observation_probs):
    n_components = start_probs.shape[0]
    delta = np.zeros((len(x), n_components))
    delta[0, :] = start_probs[:] * observation_probs[:, x[0]]

    for t in range(1, len(x), 1):
        token = x[t]
        for i in range(n_components):
            delta[t, i] = np.max(
                delta[t - 1, :] * transition_probs[:, i]
            ) * observation_probs[i, token]

    return delta

@nb.jit(nopython=True)
def _psi(x, start_probs, transition_probs, delta):
    n_components = start_probs.shape[0]
    psi = np.zeros((len(x), n_components))
    psi[0, :] = np.zeros((n_components))

    for t in range(1, len(x), 1):
        for i in range(n_components):
            psi[t, i] = np.argmax(delta[t - 1] * transition_probs[:, i])

    return psi

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
def _predict(X, lengths, start_probs, transition_probs, observation_probs):
    R = len(lengths)
    y = np.zeros(X.shape, dtype=np.int8)

    for r in nb.prange(R):
        x = X[lengths[0:r].sum():lengths[0:r+1].sum()]

        delta = _delta(x, start_probs, transition_probs, observation_probs)
        psi = _psi(x, start_probs, transition_probs, delta)

        path = np.zeros((len(x)), dtype=np.int8)
        path[-1] = np.argmax(delta[-1])

        for t in range(len(x) - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        y[lengths[0:r].sum():lengths[0:r+1].sum()] = path

    return y

@nb.jit(nopython=True, parallel=True)
def _init(X, y, lengths):
    begins = y[np.cumsum(lengths)[0:-1]]
    begins = np.hstack((y[0:1], begins))

    y_counts = np.zeros((y.max() + 1))
    yy_counts = np.zeros((y.max() + 2, y.max() + 2))
    yX_counts = np.zeros((y.max() + 1, X.max() + 1))

    for i in range(len(y)):
        y_counts[y[i]] += 1
    
    for i in range(len(y)):
        if i in begins:
            yy_counts[y.max() + 1, y[i]] += 1
        else:
            yy_counts[y[i-1], y[i]] += 1

    for i in range(len(y)):
        yX_counts[y[i], X[i]] += 1

    n_components = y.max() + 1
    n_vocabulary = X.max() + 1
    start_probs = np.zeros((n_components))
    transition_probs = np.zeros((n_components, n_components))
    observation_probs = np.zeros((n_components, n_vocabulary))

    for i in nb.prange(n_components):
        start_probs[i] = yy_counts[y.max() + 1, i] / len(lengths)

    for i in nb.prange(n_components):
        for j in nb.prange(n_components):
            transition_probs[i, j] = yy_counts[i, j] / y_counts[i]

    for i in nb.prange(n_components):
        for j in nb.prange(n_vocabulary):
            observation_probs[i, j] = yX_counts[i, j] / y_counts[i]

    return (start_probs, transition_probs, observation_probs)

@nb.jit(nopython=True, parallel=True)
def _fit(X, lengths, n_iter, start_probs, transition_probs, observation_probs):
    n_components = start_probs.shape[0]
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
        n_iter=10,
        end_prob=1.0,
        start_probs=None,
        transition_probs=None,
        observation_probs=None,
    ):
        self.n_iter = n_iter
        self.end_prob = end_prob
        self.start_probs = start_probs
        self.transition_probs = transition_probs
        self.observation_probs = observation_probs

    def score(self, X, lengths):
        return _score(
            X,
            lengths,
            self.start_probs,
            self.transition_probs,
            self.observation_probs
        )

    def predict(self, X, lengths):
        return _predict(
            X,
            lengths,
            self.start_probs,
            self.transition_probs,
            self.observation_probs
        )

    def fit(self, X, y, lengths):
        if None in [self.start_probs, self.transition_probs, self.observation_probs]:
            (self.start_probs, self.transition_probs, self.observation_probs) = _init(
                X, y, lengths
            )
            
        (self.start_probs, self.transition_probs, self.observation_probs) = _fit(
            X,
            lengths,
            self.n_iter,
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
        n_iter=5,
        start_probs=start_probs,
        transition_probs=transition_probs,
        observation_probs=observation_probs.T,
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
