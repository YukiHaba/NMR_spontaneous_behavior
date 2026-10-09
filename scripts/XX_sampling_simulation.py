# Author: Yuki Haba, 2026
"""Subsampling simulations on a fully tested colony: how many dyads and trials are needed."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils import bradley_terry, win_matrix

REFERENCE = "tube_test_reference.csv"  # fully tested colony; same columns as tube_test.csv
ANIMALS = "animals_reference.csv"      # columns: animal, weight, breeder (True/False)
                                       ## in this study, colony B: 18 animals, 153 dyads x 10 trials
N_SIM = 1000             ## simulations per condition
N_ANCHORS = 5            ## weight-stratified anchor animals
SESSION_DYADS = 8        ## dyads per testing session

ids, W_full = win_matrix(pd.read_csv(REFERENCE))
meta = pd.read_csv(ANIMALS).set_index("animal").loc[ids]
weight, breeder = meta.weight.to_numpy(float), meta.breeder.to_numpy(bool)
n = len(ids)
full_rank = pd.Series(bradley_terry(W_full)).rank(ascending=False, method="first")
all_pairs = list(combinations(range(n), 2))


def rho(theta):
    return spearmanr(full_rank, pd.Series(theta).rank(ascending=False, method="first")).statistic


class Colony:
    """Draws trials without replacement from each dyad's observed outcomes."""
    def __init__(self, rng, max_trials, sweep_stop):
        self.max_trials, self.sweep_stop = max_trials, sweep_stop  # e.g. 5 and 3: stop after 3-0, else up to 5
        self.outcomes = {(i, j): rng.permutation([1] * int(W_full[i, j]) + [0] * int(W_full[j, i]))
                         for i, j in all_pairs}
        self.W, self.theta = np.zeros((n, n)), np.zeros(n)
        self.tested, self.history = set(), []  # history: rank correlation after each dyad

    def test(self, i, j):
        i, j = min(i, j), max(i, j)
        if (i, j) in self.tested:
            return
        wins = 0
        for t, o in enumerate(self.outcomes[(i, j)][:self.max_trials], 1):
            wins += o
            if self.sweep_stop and t >= self.sweep_stop and wins in (0, t):
                break
        self.W[i, j] += wins
        self.W[j, i] += t - wins
        self.tested.add((i, j))
        self.theta = bradley_terry(self.W)
        self.history.append(rho(self.theta))


def adaptive(rng, max_trials, sweep_stop, n_refine):
    """Weight-guided adaptive testing (see Methods)."""
    c = Colony(rng, max_trials, sweep_stop)
    nonbreeders = np.flatnonzero(~breeder)
    # 1. round robin among weight-stratified anchors
    by_weight = nonbreeders[np.argsort(weight[nonbreeders])]
    anchors = sorted(set(by_weight[np.round(np.linspace(0, len(by_weight) - 1, N_ANCHORS)).astype(int)]))
    for i, j in combinations(anchors, 2):
        c.test(i, j)
    # 2. place each remaining nonbreeder: test against the animal at its weight-predicted rank, then bracket
    placed = list(anchors)
    for a in rng.permutation([x for x in nonbreeders if x not in anchors]):
        order = sorted(placed, key=lambda x: -c.theta[x])
        if len(placed) >= 4:
            fit = np.polyfit(weight[order], np.arange(1, len(order) + 1), 1)
            predicted = np.clip(np.polyval(fit, weight[a]), 1, len(order))
        else:
            predicted = (len(order) + 1) / 2
        placed.append(a)
        c.test(a, order[int(np.clip(round(predicted) - 1, 0, len(order) - 1))])
        for _ in range(2):  # up to two bracketing dyads with untested neighbors
            order = sorted(placed, key=lambda x: -c.theta[x])
            k = order.index(a)
            neighbors = [order[m] for m in (k - 1, k + 1) if 0 <= m < len(order)]
            untested = [b for b in neighbors if (min(a, b), max(a, b)) not in c.tested]
            if not untested:
                break
            c.test(a, untested[0])
    # 3. breeders: test against the top- and middle-ranked nonbreeders
    for a in np.flatnonzero(breeder):
        order = sorted(placed, key=lambda x: -c.theta[x])
        for b in (order[0], order[len(order) // 2]):
            c.test(a, b)
    # 4. refinement: untested neighboring ranks with the smallest score gaps
    order = np.argsort(-c.theta)
    gaps = sorted((abs(c.theta[x] - c.theta[y]), x, y) for x, y in zip(order[:-1], order[1:])
                  if (min(x, y), max(x, y)) not in c.tested)
    for _, x, y in gaps[:n_refine]:
        c.test(x, y)
    return c


def extend(c, coverage, rng):
    """Add untested dyads with the smallest score gaps until the target coverage."""
    target = round(len(all_pairs) * coverage)
    candidates = [p for p in rng.permutation(all_pairs).tolist() if tuple(p) not in c.tested]
    candidates.sort(key=lambda p: abs(c.theta[p[0]] - c.theta[p[1]]))
    for i, j in candidates[:max(0, target - len(c.tested))]:
        c.test(i, j)


def summary(values):
    return pd.Series({"mean": np.mean(values), "q10": np.quantile(values, 0.1), "q90": np.quantile(values, 0.9)})


rng = np.random.default_rng(0)
# (A) trials per dyad at fixed dyad coverage
COVERAGE_A = 0.4  ## in this study, 40% of dyads
rows = {}
for label, max_trials, sweep in [(str(k), k, None) for k in range(1, 11)] + [("3-0 or 5", 5, 3)]:
    out = []
    for _ in range(N_SIM):
        c = adaptive(rng, max_trials, sweep, n_refine=2)
        extend(c, COVERAGE_A, rng)
        out.append(c.history[-1])
    rows[label] = summary(out)
print(pd.DataFrame(rows).T)

# (B) dyad coverage with the 3-0-or-5 rule
N_SESSIONS = 6                    ## in this study, first 6 sessions
COVERAGE_B = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sessions, points, coverage_adaptive = [], {cov: [] for cov in ["adaptive"] + COVERAGE_B}, []
for _ in range(N_SIM):
    # progression across testing sessions
    c = adaptive(rng, 5, 3, n_refine=6)
    ends = [min(k * SESSION_DYADS, len(c.history)) - 1 for k in range(1, N_SESSIONS + 1)]
    sessions.append([c.history[e] for e in ends])
    # end of adaptive testing, then additional dyads with the smallest score gaps
    c = adaptive(rng, 5, 3, n_refine=2)
    coverage_adaptive.append(len(c.tested) / len(all_pairs))
    points["adaptive"].append(c.history[-1])
    for cov in COVERAGE_B:
        extend(c, cov, rng)
        points[cov].append(c.history[-1])
print(pd.DataFrame([summary(s) for s in np.array(sessions).T], index=range(1, N_SESSIONS + 1)))
print("coverage after adaptive testing:", np.mean(coverage_adaptive))
print(pd.DataFrame({k: summary(v) for k, v in points.items()}).T)
