# Author: Yuki Haba, 2026
"""UMAP embedding of syllable instances."""
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from utils import runs

RESULTS = "results.h5"
SYLLABLES = "syllables.csv"
MIN_FRAMES = 10  ## shortest instance embedded
N_BINS = 25      ## time bins per instance


def resample(x, n):
    x = np.asarray(x, float).reshape(len(x), -1)
    t, tq = np.linspace(0, 1, len(x)), np.linspace(0, 1, n)
    return np.column_stack([np.interp(tq, t, col) for col in x.T]).ravel()


def instance_vector(latent, xy, heading):
    """Latent trajectory + forward/lateral displacement + heading change."""
    step, theta = np.diff(xy, axis=0), heading[:-1]
    forward = np.r_[0, np.cumsum(step[:, 0] * np.cos(theta) + step[:, 1] * np.sin(theta))]
    lateral = np.r_[0, np.cumsum(-step[:, 0] * np.sin(theta) + step[:, 1] * np.cos(theta))]
    turn = np.unwrap(heading) - np.unwrap(heading)[0]
    return np.concatenate([resample(v, N_BINS) for v in (latent, forward, lateral, turn)])


syl = pd.read_csv(SYLLABLES)
family = dict(zip(syl.syllable[syl.confidence == "high"], syl.family[syl.confidence == "high"]))

rows, X = [], []
with h5py.File(RESULTS, "r") as h:
    for rec in h:  ## in this study, all recordings in the model
        g = {k: h[rec][k][()] for k in ("syllable", "latent_state", "centroid", "heading")}
        for s, e, lab in runs(g["syllable"]):
            if lab in family and e - s >= MIN_FRAMES:
                X.append(instance_vector(g["latent_state"][s:e], g["centroid"][s:e], g["heading"][s:e]))
                rows.append({"recording": rec, "syllable": lab, "family": family[lab], "n_frames": e - s})

embedding = UMAP(n_neighbors=30, min_dist=0.2, metric="euclidean", random_state=123) \
    .fit_transform(StandardScaler().fit_transform(np.array(X)))
pd.DataFrame(rows).assign(umap_1=embedding[:, 0], umap_2=embedding[:, 1]).to_csv("umap.csv", index=False)
