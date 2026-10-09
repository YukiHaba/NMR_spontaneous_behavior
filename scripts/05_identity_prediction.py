# Author: Yuki Haba, 2026
"""Individual identity predicted from short windows of each animal's recording."""
import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from utils import FPS, feature_columns, random_forest, syllable_features

RESULTS = "results.h5"
SYLLABLES = "syllables.csv"
ANIMALS = "animals.csv"
RECORDING_S = 600  ## recording length (s); in this study, 10 min

syl = pd.read_csv(SYLLABLES)
syllables = sorted(syl.syllable[syl.confidence == "high"])
animals = pd.read_csv(ANIMALS)


def windows(window_s):
    """Syllable features for consecutive non-overlapping windows of each recording."""
    rows = []
    with h5py.File(RESULTS, "r") as h:
        for a in animals.itertuples():
            labels, heading = h[a.recording]["syllable"][()], h[a.recording]["heading"][()]
            for start in range(0, RECORDING_S - window_s + 1, window_s):
                i, j = start * FPS, (start + window_s) * FPS
                rows.append({"colony": a.colony, "animal": a.animal, "start": start, "end": start + window_s,
                             **syllable_features(labels[i:j], heading[i:j], syllables)})
    return pd.DataFrame(rows)


def scan(w, block_s, block="test"):
    """Slide a contiguous block across the recording in one-window steps.
    The block is the test set (block="test") or the training set (block="train");
    windows crossing the block boundary are excluded."""
    step = int(w.end[0] - w.start[0])
    out = []
    for colony, d in w.groupby("colony"):
        X = d[feature_columns(d)].fillna(0)
        for b0 in range(0, RECORDING_S - block_s + 1, step):
            inside = (d.start >= b0) & (d.end <= b0 + block_s)
            outside = (d.end <= b0) | (d.start >= b0 + block_s)
            train, test = (outside, inside) if block == "test" else (inside, outside)
            rf = random_forest(n_trees=20, max_depth=6, seed=123 + (b0 + block_s) // step).fit(X[train], d.animal[train])
            out.append(pd.DataFrame({"colony": colony, "true": d.animal[test], "predicted": rf.predict(X[test])}))
    return pd.concat(out)


def recovered(pred):
    """Per colony: animals whose true identity is the single most frequent prediction,
    tested against 1/N chance (one-sided exact binomial, Bonferroni across colonies)."""
    res = []
    for colony, d in pred.groupby("colony"):
        counts = d.groupby(["true", "predicted"]).size().unstack(fill_value=0)
        top = [row.get(a, 0) == row.max() and (row == row.max()).sum() == 1 for a, row in counts.iterrows()]
        n, k = len(top), int(np.sum(top))
        res.append({"colony": colony, "recovered": k, "n_animals": n,
                    "p": binomtest(k, n, 1 / n, alternative="greater").pvalue})
    res = pd.DataFrame(res)
    res["p_bonferroni"] = np.minimum(1, res.p * len(res))
    return res


# primary analysis: 30-s windows, 5-min test block
w30 = windows(30)
print(recovered(scan(w30, block_s=300)))

# sensitivity: training/test duration (30-s windows)
for test_min in [0.5, 1, 2, 3, 4]:  ## in this study, 9.5:0.5 to 6:4 min (train:test)
    print(test_min, recovered(scan(w30, block_s=int(test_min * 60))))

# sensitivity: window size (5-min training and test)
for window_s, block in [(15, "train"), (45, "train"), (60, "test")]:  ## in this study, 15, 30, 45, 60 s
    print(window_s, recovered(scan(windows(window_s), block_s=300, block=block)))
