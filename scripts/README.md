# Scripts

Run in numerical order from a folder containing the input files below. `XX_sampling_simulation.py` was used to design tube testing and is run separately. Lines marked `##` are settings; the value used in this study is given there.

| Script | Reads | Writes |
|---|---|---|
| `01_fit_kpms.py` | SLEAP files, `config/kpms_config.yml` | Keypoint-MoSeq models; final model `results.h5` |
| `02_syllable_features.py` | `results.h5`, `syllables.csv`, `animals.csv` | `animal_features.csv`, `family_transitions.csv`, `family_frequency.csv` |
| `03_umap.py` | `results.h5`, `syllables.csv` | `umap.csv` |
| `04_colony_prediction.py` | `animal_features.csv` | `colony_confusion.csv` |
| `05_identity_prediction.py` | `results.h5`, `syllables.csv`, `animals.csv` | printed results |
| `06_sex_status_prediction.py` | `animal_features.csv` | `sex_confusion.csv` |
| `07_dominance_rank.py` | `tube_test.csv` | `dominance_ranks.csv` |
| `08_rank_prediction.py` | `animal_features.csv` | printed results |
| `XX_sampling_simulation.py` | `tube_test_reference.csv`, `animals_reference.csv` | printed results |

## Input files

- `syllables.csv`: `syllable, family, confidence` (`high` for syllables used in analyses)
- `animals.csv`: `recording, animal, colony, sex, reproductive_role, rank` (one recording per animal; `rank` from `07`, 1 = most dominant)
- `tube_test.csv`: `colony, time_point, animal_a, animal_b, wins_a, wins_b`
- `tube_test_reference.csv`: same columns, for the fully tested colony
- `animals_reference.csv`: `animal, weight, breeder`
