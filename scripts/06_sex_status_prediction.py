# Author: Yuki Haba, 2026
"""Sex and reproductive status predicted from each animal's syllable profile."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from utils import feature_columns, holm, random_forest

N_PERM = 1000  ## label permutations
BREEDERS = ["queen", "breeding_male"]

d = pd.read_csv("animal_features.csv")  # from 02_syllable_features.py
FEATURES = feature_columns(d)
rng = np.random.default_rng(0)


def loao(X, y):
    return cross_val_predict(random_forest(n_trees=50, max_depth=8), X, y, cv=LeaveOneOut())


def test(df, label, groups):
    """Observed balanced accuracy and one-sided permutation P (labels shuffled within colony,
    full leave-one-animal-out analysis repeated for every permutation)."""
    X, y = df[FEATURES].fillna(0), df[label].to_numpy()
    pred = loao(X, y)
    observed = balanced_accuracy_score(y, pred)
    null = []
    for _ in range(N_PERM):
        y_perm = y.copy()
        for g in np.unique(groups):
            y_perm[groups == g] = rng.permutation(y[groups == g])
        null.append(balanced_accuracy_score(y_perm, loao(X, y_perm)))
    return observed, (1 + np.sum(np.array(null) >= observed)) / (N_PERM + 1), pred


# sex: separate classifiers per colony, with and without breeders
sexed = d[d.sex.isin(["F", "M"])]  ## in this study, 82 animals with known sex
results, pooled = [], []
for breeders in [True, False]:
    for colony, df in sexed.groupby("colony"):
        df = df if breeders else df[~df.reproductive_role.isin(BREEDERS)]
        acc, p, pred = test(df, "sex", df.colony.to_numpy())
        results.append([colony, breeders, acc, p])
        if breeders:
            pooled.append(pd.DataFrame({"true": df.sex, "predicted": pred}))
sex = pd.DataFrame(results, columns=["colony", "with_breeders", "balanced_accuracy", "p"])
sex["p_holm"] = holm(sex.p)
print(sex)
pooled = pd.concat(pooled)
pd.crosstab(pooled.true, pooled.predicted).to_csv("sex_confusion.csv")  # predictions pooled across colonies

# reproductive status: all animals, three classes and breeder vs nonbreeder
d["breeder"] = d.reproductive_role.isin(BREEDERS)
status = pd.DataFrame([test(d, label, d.colony.to_numpy())[:2] for label in ["reproductive_role", "breeder"]],
                      index=["three_class", "binary"], columns=["balanced_accuracy", "p"])
status["p_holm"] = holm(status.p)
print(status)
