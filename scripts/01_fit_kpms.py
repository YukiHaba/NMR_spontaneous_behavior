# Author: Yuki Haba, 2026
"""Fit Keypoint-MoSeq models and select the final model."""
import jax
import numpy as np
import keypoint_moseq as kpms

PROJECT = "kpms_project"        # folder with config.yml (see config/kpms_config.yml)
SLEAP_FILES = "sleap/*.slp"     # SLEAP pose tracks, one file per recording
AR_ITERS = 50                   ## AR-HMM initialization iterations
KAPPA_SCAN = [1e5, 2e5, 5e5, 1e6, 2e6, 3e6, 5e6, 7e6, 1e7]  ## kappa values compared
SCAN_ITERS = 250                ## total iterations per kappa-scan model
KAPPA = 1e7                     ## selected kappa
N_SEEDS = 30                    ## random initializations at the selected kappa
FINAL_ITERS = 500               ## total iterations per seed model

config = kpms.load_config(PROJECT)
coordinates, confidences, _ = kpms.load_keypoints(SLEAP_FILES, "sleap")
## recordings trimmed to the 10-min analysis window before this step
coordinates, confidences = kpms.outlier_removal(coordinates, confidences, PROJECT, **config)
data, metadata = kpms.format_data(coordinates, confidences, **config)
pca = kpms.fit_pca(**data, **config)
kpms.save_pca(pca, PROJECT)


def fit(name, kappa, seed, total_iters):
    model = kpms.init_model(data, pca=pca, **config, seed=jax.random.PRNGKey(seed))
    model = kpms.update_hypparams(model, kappa=kappa)
    model, _ = kpms.fit_model(model, data, metadata, PROJECT, name, ar_only=True, num_iters=AR_ITERS)
    model = kpms.update_hypparams(model, kappa=kappa)
    kpms.fit_model(model, data, metadata, PROJECT, name, ar_only=False,
                   start_iter=AR_ITERS, num_iters=total_iters)
    kpms.reindex_syllables_in_checkpoint(PROJECT, name, runlength=False)  # number syllables by frequency
    model, _, metadata_, _ = kpms.load_checkpoint(PROJECT, name)
    kpms.extract_results(model, metadata_, PROJECT, name)


# 1. kappa scan; kappa chosen from syllable durations, usage, trajectories and videos
for kappa in KAPPA_SCAN:
    fit(f"kappa{kappa:.0e}", kappa, 0, SCAN_ITERS)

# 2. random initializations at the chosen kappa; keep the highest expected marginal likelihood
names = [f"kappa{KAPPA:.0e}_seed{seed:02d}" for seed in range(N_SEEDS)]
for seed, name in enumerate(names):
    fit(name, KAPPA, seed, FINAL_ITERS)
scores, _ = kpms.expected_marginal_likelihoods(PROJECT, names)
print("final model:", names[int(np.argmax(scores))])
