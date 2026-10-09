# Author: Yuki Haba, 2026
"""Shared helpers."""
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier

FPS = 60  ## frames per second; in this study, 60
SUMMARY = ["mean_abs_angular_velocity", "n_syllables_observed",
           "mean_syllable_duration", "max_syllable_duration", "n_unique_transitions"]


def runs(labels):
    """Contiguous runs of the same label as (start, end, label)."""
    labels = np.asarray(labels)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    return [(s, e, labels[s]) for s, e in zip(starts, ends)]


def syllable_features(labels, heading, syllables):
    """Fraction of frames in each syllable + 5 summary statistics."""
    labels = np.asarray(labels)
    valid = labels[labels >= 0]
    f = {f"syl_{s}": np.mean(valid == s) for s in syllables}
    changes = valid[1:] != valid[:-1]
    durations = [(e - s) / FPS for s, e, lab in runs(labels) if lab >= 0]
    f["mean_abs_angular_velocity"] = np.nanmean(np.abs(np.diff(np.unwrap(heading)))) * FPS
    f["n_syllables_observed"] = len(np.unique(valid))
    f["mean_syllable_duration"] = np.mean(durations)
    f["max_syllable_duration"] = np.max(durations)
    f["n_unique_transitions"] = len(set(zip(valid[:-1][changes], valid[1:][changes])))
    return f


def feature_columns(df):
    """Syllable fractions + summary statistics (in this study, 51 + 5 = 56 features)."""
    return [c for c in df.columns if c.startswith("syl_")] + SUMMARY


def random_forest(n_trees, max_depth, seed=123):
    return RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                  class_weight="balanced_subsample", random_state=seed)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def win_matrix(df):
    """Tube-test wins: W[i, j] = trials won by animal i against animal j."""
    ids = sorted(set(df.animal_a) | set(df.animal_b))
    idx = {a: i for i, a in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)))
    for r in df.itertuples():
        W[idx[r.animal_a], idx[r.animal_b]] += r.wins_a
        W[idx[r.animal_b], idx[r.animal_a]] += r.wins_b
    return ids, W


def bradley_terry(W, lam=0.1):
    """Ridge-penalized Bradley-Terry scores (lambda = 0.1); last animal fixed at 0, then centered."""
    i, j = np.nonzero(np.triu(W + W.T))
    w, total = W[i, j], W[i, j] + W[j, i]

    def loss(x):
        theta = np.r_[x, 0]
        diff = theta[i] - theta[j]
        value = -np.sum(w * diff - total * np.logaddexp(0, diff)) + lam * np.sum(theta ** 2)
        g = total / (1 + np.exp(-diff)) - w
        grad = np.bincount(i, g, len(theta)) - np.bincount(j, g, len(theta)) + 2 * lam * theta
        return value, grad[:-1]

    theta = np.r_[minimize(loss, np.zeros(len(W) - 1), jac=True, method="L-BFGS-B").x, 0]
    return theta - theta.mean()
