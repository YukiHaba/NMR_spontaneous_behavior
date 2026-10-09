# Author: Yuki Haba, 2026
"""Bradley-Terry dominance ranks, hierarchy transitivity and rank stability."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils import bradley_terry, win_matrix

TUBE_TEST = "tube_test.csv"  # columns: colony, time_point, animal_a, animal_b, wins_a, wins_b
N_PERM = 10_000              ## permutations for transitivity


def transitivity(W):
    """Fraction of complete, untied triads without a dominance cycle."""
    D = W > W.T                     # dyad winner (more trials won)
    tested = (W + W.T > 0) & (W != W.T)
    n_complete = n_transitive = 0
    for a, b, c in combinations(range(len(W)), 3):
        if tested[a, b] and tested[a, c] and tested[b, c]:
            n_complete += 1
            cycle = (D[a, b] and D[b, c] and D[c, a]) or (D[b, a] and D[c, b] and D[a, c])
            n_transitive += not cycle
    return n_transitive / n_complete


def transitivity_test(W, rng):
    """Null: trials per dyad fixed, wins redrawn with equal win probability."""
    observed = transitivity(W)
    n = np.triu(W + W.T).astype(int)
    null = []
    for _ in range(N_PERM):
        wins = rng.binomial(n, 0.5)
        null.append(transitivity(wins + (n - wins).T))
    return observed, (1 + np.sum(np.array(null) >= observed)) / (N_PERM + 1)


tt = pd.read_csv(TUBE_TEST)
rng = np.random.default_rng(7)
ranks = []
for (colony, time_point), df in tt.groupby(["colony", "time_point"]):
    ids, W = win_matrix(df)
    theta = bradley_terry(W)
    rank = pd.Series(theta, ids).rank(ascending=False, method="first")
    ranks.append(pd.DataFrame({"colony": colony, "time_point": time_point, "animal": ids,
                               "score": theta, "rank": rank.to_numpy()}))
    # dyads won by the higher-ranked animal (expected), by the lower-ranked animal (upset), or tied
    r = rank.to_numpy()
    hi = np.where(r[:, None] < r[None, :], W, W.T)
    lo = np.where(r[:, None] < r[None, :], W.T, W)
    iu = np.triu_indices(len(W), 1)
    tested = (W + W.T)[iu] > 0
    outcome = np.sign(hi[iu] - lo[iu])[tested]
    score, p = transitivity_test(W, rng)
    print(colony, time_point, f"expected {np.sum(outcome > 0)}, upsets {np.sum(outcome < 0)}, ties {np.sum(outcome == 0)};",
          f"transitivity {score:.3f}, P = {p:.2g}")
ranks = pd.concat(ranks)
ranks.to_csv("dominance_ranks.csv", index=False)

# rank stability: ranks of animals tested at both time points (re-ranked among shared animals)
for colony, df in ranks.groupby("colony"):
    if df.time_point.nunique() < 2:
        continue
    wide = df.pivot(index="animal", columns="time_point", values="score").dropna()
    rho, p = spearmanr(wide.iloc[:, 0], wide.iloc[:, 1])
    print(colony, f"rank stability: n = {len(wide)}, rho = {rho:.2f}, P = {p:.2g}")
