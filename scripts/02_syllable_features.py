# Author: Yuki Haba, 2026
"""Per-animal syllable features and syllable-family transitions."""
import h5py
import pandas as pd
from utils import runs, syllable_features

RESULTS = "results.h5"      # final Keypoint-MoSeq model (01_fit_kpms.py)
SYLLABLES = "syllables.csv"  # columns: syllable, family, confidence
ANIMALS = "animals.csv"      # columns: recording, animal, colony, sex, reproductive_role, rank
                             ## one recording per animal; in this study, 85 animals

syl = pd.read_csv(SYLLABLES)
high = syl[syl.confidence == "high"]
syllables = sorted(high.syllable)  ## in this study, 51 high-confidence syllables in 7 families
family = dict(zip(high.syllable, high.family))
animals = pd.read_csv(ANIMALS)

rows, transitions = [], []
with h5py.File(RESULTS, "r") as h:
    for a in animals.itertuples():
        labels, heading = h[a.recording]["syllable"][()], h[a.recording]["heading"][()]
        rows.append({"recording": a.recording, **syllable_features(labels, heading, syllables)})
        # family of each syllable instance; non-high-confidence instances break the sequence
        fams = [family.get(lab) for _, _, lab in runs(labels)]
        transitions += [(a.colony, f1, f2) for f1, f2 in zip(fams[:-1], fams[1:]) if f1 and f2]

features = animals.merge(pd.DataFrame(rows), on="recording")
features.to_csv("animal_features.csv", index=False)

# transition probabilities between syllable families, pooled over animals
t = pd.DataFrame(transitions, columns=["colony", "from_family", "to_family"])
t = pd.concat([t, t.assign(colony="all")])
counts = t.groupby(["colony", "from_family", "to_family"]).size().rename("n").reset_index()
counts["probability"] = counts.n / counts.groupby(["colony", "from_family"]).n.transform("sum")
counts.to_csv("family_transitions.csv", index=False)

# family frequency: fraction of frames in each family, averaged over animals
frac = pd.DataFrame({f: features[[f"syl_{s}" for s in syllables if family[s] == f]].sum(axis=1)
                     for f in high.family.unique()})
frac["colony"] = features.colony
pd.concat([frac, frac.assign(colony="all")]).groupby("colony").mean().to_csv("family_frequency.csv")
