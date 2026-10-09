# Author: Yuki Haba, 2026
"""Dominance rank predicted from syllable profiles (PLS regression), within and across colonies."""
from itertools import combinations, product
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from utils import feature_columns, holm

N_COMPONENTS = {"B": 1, "H": 3, "P": 4, "S": 1}  ## PLS components per colony; in this study, B 1, H 3, P 4, S 1
N_PERM = 10_000                                   ## permutations for each cross-colony prediction

d = pd.read_csv("animal_features.csv").dropna(subset=["rank"])  # rank: 1 = most dominant
## in this study, B 15, H 18, P 24, S 19 ranked animals
FEATURES = feature_columns(d)
colonies = {c: (g[FEATURES].fillna(0).to_numpy(), g["rank"].to_numpy()) for c, g in d.groupby("colony")}


def fit_predict(X_train, rank_train, X_test, k):
    """Standardize on training animals, scale ranks from 0.5 (highest) to -0.5 (lowest), fit PLS."""
    scaler = StandardScaler().fit(X_train)
    y = 0.5 - (rankdata(rank_train) - 1) / (len(rank_train) - 1)
    pls = PLSRegression(n_components=k, scale=False).fit(scaler.transform(X_train), y)
    return pls.predict(scaler.transform(X_test)).ravel()


def evaluate(rank, pairs, scores):
    """Pairwise accuracy and rank order reconstructed from pairwise wins."""
    higher = scores[:, 0] > scores[:, 1]
    correct = higher == (rank[pairs[:, 0]] < rank[pairs[:, 1]])
    wins = np.zeros(len(rank))
    np.add.at(wins, pairs[:, 0], higher)
    np.add.at(wins, pairs[:, 1], ~higher)
    return int(correct.sum()), spearmanr(rank, rankdata(-wins)).statistic


# within colony: leave-two-animals-out
within = {}
for c, (X, rank) in colonies.items():
    pairs = np.array(list(combinations(range(len(rank)), 2)))
    scores = np.array([fit_predict(np.delete(X, p, 0), np.delete(rank, p), X[p], N_COMPONENTS[c]) for p in pairs])
    n_correct, rho = evaluate(rank, pairs, scores)
    p = binomtest(n_correct, len(pairs), 0.5, alternative="greater").pvalue
    within[c] = (n_correct / len(pairs), rho)
    print(f"{c}: {n_correct}/{len(pairs)} pairs, Bonferroni P = {min(1, p * len(colonies)):.2g}; rank rho = {rho:.2f}")

# across colonies: fit on all animals of the source colony, predict every other colony
rng = np.random.default_rng(0)
rows = []
for source, target in product(colonies, colonies):
    if source == target:
        rows.append([source, target, *within[source], np.nan])
        continue
    (Xs, rs), (Xt, rt) = colonies[source], colonies[target]
    scores = fit_predict(Xs, rs, Xt, N_COMPONENTS[source])
    pairs = np.array(list(combinations(range(len(rt)), 2)))
    n_correct, rho = evaluate(rt, pairs, scores[pairs])
    null = [spearmanr(rng.permutation(rt), -scores).statistic for _ in range(N_PERM)]
    rows.append([source, target, n_correct / len(pairs), rho, (1 + np.sum(np.array(null) >= rho)) / (N_PERM + 1)])
t = pd.DataFrame(rows, columns=["source", "target", "accuracy", "rho", "p"])
cross = t.source != t.target
t.loc[cross, "p_holm"] = holm(t.p[cross])
print(t)

# within- vs cross-colony means: exact test over all ways of picking one "within" result
# per source colony (in this study, 4^4 = 256; two-sided)
for metric in ["accuracy", "rho"]:
    by_source = [t[t.source == c][metric].to_numpy() for c in colonies]
    observed = t[metric][~cross].mean() - t[metric][cross].mean()
    diffs = []
    for pick in product(*[range(len(r)) for r in by_source]):
        chosen = np.array([r[i] for r, i in zip(by_source, pick)])
        rest = np.concatenate([np.delete(r, i) for r, i in zip(by_source, pick)])
        diffs.append(chosen.mean() - rest.mean())
    print(f"{metric}: within {t[metric][~cross].mean():.2f} vs cross {t[metric][cross].mean():.2f}, "
          f"P = {np.mean(np.abs(diffs) >= abs(observed) - 1e-12):.3g}")
