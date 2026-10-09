# Author: Yuki Haba, 2026
"""Colony membership predicted from each animal's syllable profile."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from utils import feature_columns, random_forest

N_PERM = 200_000  ## label permutations

d = pd.read_csv("animal_features.csv")  # from 02_syllable_features.py
X, y = d[feature_columns(d)].fillna(0), d.colony.to_numpy()

# leave-one-animal-out random forest
pred = cross_val_predict(random_forest(n_trees=50, max_depth=8), X, y, cv=LeaveOneOut())
accuracy = np.mean(pred == y)

# overall significance: permute colony labels against the held-out predictions
rng = np.random.default_rng(0)
null = np.array([np.mean(rng.permutation(y) == pred) for _ in range(N_PERM)])
p_perm = (1 + np.sum(null >= accuracy)) / (N_PERM + 1)
print(f"accuracy {accuracy:.3f}, permutation P = {p_perm:.2g}")

# per colony: one-sided exact binomial test vs 1/(number of colonies), Bonferroni
colonies = np.unique(y)  ## in this study, 4 colonies
for c in colonies:
    k, n = int(np.sum(pred[y == c] == c)), int(np.sum(y == c))
    p = binomtest(k, n, 1 / len(colonies), alternative="greater").pvalue
    print(f"{c}: {k}/{n}, Bonferroni P = {min(1, p * len(colonies)):.2g}")

pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="predicted")).to_csv("colony_confusion.csv")
