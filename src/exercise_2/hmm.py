import re as re

import numba as nb
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

import sklearn.base as skbase


@nb.jit(nopython=True, fastmath=True)
def _alpha(x, n_components, start_probs, transition_probs, observation_probs):
    alpha = np.zeros((len(x), n_components))
    alpha[0, :] = start_probs[:] * observation_probs[:, x[0]]

    for t in range(1, len(x), 1):
        token = x[t]
        for i in range(n_components):
            alpha[t, i] = observation_probs[i, token] * sum([
                alpha[t - 1, j] * transition_probs[j, i]
                for j in range(n_components)
            ])

    return alpha


@nb.jit(nopython=True, fastmath=True)
def _beta(x, n_components, start_probs, transition_probs, observation_probs):
    beta = np.zeros((len(x), n_components))
    beta[-1, :] = np.ones((1, n_components))

    for t in range(len(x) - 2, -1, -1):
        next_token = x[t + 1]
        for i in range(n_components):
            beta[t, i] = sum([
                beta[t + 1, j]
                * transition_probs[i, j]
                * observation_probs[j, next_token]
                for j in range(n_components)
            ])

    return beta


@nb.jit(nopython=True, fastmath=True)
def _xi(x, alpha, beta, n_components, start_probs, transition_probs, observation_probs):
    xi = np.zeros((len(x) - 1, n_components, n_components))

    for t in range(0, len(x) - 1, 1):
        next_token = x[t + 1]
        for i in range(n_components):
            for j in range(n_components):
                xi[t, i, j] = (
                    alpha[t, i]
                    * transition_probs[i, j]
                    * beta[t + 1, j]
                    * observation_probs[j, next_token]
                ) / sum([
                    alpha[t, k]
                    * transition_probs[k, w]
                    * beta[t + 1, w]
                    * observation_probs[w, next_token]
                    for w in range(n_components)
                    for k in range(n_components)
                ])

    return xi


@nb.jit(nopython=True, fastmath=True)
def _gamma(x, alpha, beta, n_components, start_probs, transition_probs, observation_probs):
    gamma = np.zeros((len(x), n_components))

    for t in range(0, len(x), 1):
        for i in range(n_components):
            gamma[t, i] = alpha[t, i] * beta[t, i] / sum([
                (alpha[t, j] * beta[t, j])
                for j in range(n_components)
            ])

    return gamma


@nb.jit(nopython=True, parallel=True, fastmath=True)
def _fit(X, n_iter, n_components, start_probs, transition_probs, observation_probs):
    for _ in range(n_iter):
        R = len(X)

        xis = np.zeros((R, len(X[0]) - 1, n_components, n_components))
        gammas = np.zeros((R, len(X[0]), n_components))

        for r in nb.prange(R):
            x = X[r]

            alpha = _alpha(x, n_components, start_probs,
                           transition_probs, observation_probs)
            beta = _beta(x, n_components, start_probs,
                         transition_probs, observation_probs)
            xis[r, :, :, :] = _xi(x, alpha, beta, n_components, start_probs,
                                  transition_probs, observation_probs)
            gammas[r, :, :] = _gamma(x, alpha, beta, n_components, start_probs,
                                     transition_probs, observation_probs)

            print(alpha)
            print(beta)

        for i in range(n_components):
            start_probs[i] = sum([
                gammas[r, 0, i]
                for r in range(R)
            ]) / R

        for i in range(n_components):
            for j in range(n_components):
                transition_probs[i, j] = sum([
                    xis[r, t, i, j]
                    for r in range(R)
                    for t in range(len(X[r]) - 1)
                ]) / sum([
                    gammas[r, t, i]
                    for r in range(R)
                    for t in range(len(X[r]) - 1)
                ])

        for i in range(n_components):
            for k in range(n_components):
                observation_probs[i, k] = sum([
                    gammas[r, t, i]
                    for r in range(R)
                    for t in range(len(X[r]))
                    if X[r][t] == k
                ]) / sum([
                    gammas[r, t, i]
                    for r in range(R)
                    for t in range(len(X[r]))
                ])

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

    def score(self, X, y=None):
        result = []
        for x in X:
            alpha = _alpha(x, self.n_components, self.start_probs,
                           self.transition_probs, self.observation_probs)
            result.append(sum(alpha[-1, :]))

        return result

    def predict(self, X, y=None):
        pass

    def fit(self, X, y=None):
        (self.start_probs, self.transition_probs, self.observation_probs) = _fit(
            X,
            self.n_iter,
            self.n_components,
            self.start_probs,
            self.transition_probs,
            self.observation_probs
        )


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
