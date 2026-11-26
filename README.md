# busi-quantum-bottleneck

Protocol, code and results for the study **"Does a quantum bottleneck help breast ultrasound lesion segmentation? A prespecified comparison with classical layers in the same slot"** (S. Parii, A. Tujammal, A. Iqbal, R. Ashraf, F. Aadil; submitted to *Artificial Intelligence in Medicine*).

A simulated circuit with four qubits in the bottleneck slot of a U-Net is compared, in the same slot and from the same initial weights, with five classical blocks: the same circuit without CNOT gates (Qp), a layer that spans the full Fourier series of the circuit (S), a trigonometric layer (Tr), a dense layer (Cl) and an identity mapping (Id). The analysis plan was fixed before the first training run (`protocols/PROTOCOL_busi_prespecified.txt`).

Archived version: Zenodo, https://doi.org/10.5281/zenodo.24536631

No image data are stored in this repository. BUSI, BUS-BRA and BrEaST are downloaded from their original sources (protocol, Section 2; manuscript, Data availability).

## Contents

| Folder | Content |
|---|---|
| `protocols/` | The prespecified protocol (data cleaning, folds, models, training, run plans, metrics, statistics, claims) |
| `src/` | `qsim.py` state-vector simulator; `qbus.py` models, blocks, training and bottleneck audit; `prep.py` data preparation; `runner.py` run list and Drive lock files; `analysis.py` statistics of protocol Section 7; `qnoise.py` density-matrix noise model; `noise_run.py` noise test; `qual_run.py` qualitative examples |
| `tests/` | Checks of the simulator and of the noise model against an independent implementation |
| `notebooks/` | Google Colab notebooks: `00_data_prep`, `01_train` and `01_train_B` (training, two sessions), `02_collect`, `02b_split_busbra`, `03_noise` (exploratory noise test), `04_qualitative` (examples of Fig. 6), `05_equal_k` (exploratory rerun of Qp4 and Q8 with k = 100, run in two sessions) |
| `analysis/` | Scripts that turn the result files into the tables and figures of the manuscript |
| `results/data/` | Files written by `02_collect`: split of every BUSI image (`busi_manifest.csv`), summary of every run (`summaries.csv`), Dice of every test image in every run (`test_busi`, `test_busbra`, `test_breast`), normal images, code replacement, finite shots, selected learning rate factors (`plan_decision.json`) |
| `results/stats/` | Statistics computed from `results/data/` |
| `results/noise/` | Exploratory noise test: Dice of every BUSI image for every run, noise level and number of shots |
| `results/qualitative/` | Check of the seven retrained models used for Fig. 6 |
| `results/k100/` | Exploratory rerun with equal factors: per-run summaries and per-image Dice of Qp4 and Q8 with k = 100 |

## Reproducing the tables and figures

```
python src/analysis.py results/data results/stats            # model levels, all contrasts, audit, costs (protocol Section 7)
python analysis/gap.py results/data results/stats/gap_contrasts.csv          # exploratory gap contrasts
python analysis/noise.py results/noise/noise_q4.csv.gz results/stats/noise_contrasts.csv   # exploratory noise test
python analysis/equal_k.py results/data results/k100 results/stats/equal_k_contrasts.csv   # exploratory rerun with equal factors
RESULTS=results/stats/ DECISION=results/data/plan_decision.json OUT=out python analysis/tabs2.py
RESULTS=results/stats/ OUT=out python analysis/figs2.py
```

| Manuscript | Source |
|---|---|
| Table 3 (model levels), Supplementary Table S1 | `results/stats/model_levels.csv`, `analysis/tabs2.py` |
| Table 4 (F1 and F2 contrasts), Fig. 4, Supplementary Table S2 | `results/stats/contrasts.csv`, `analysis/tabs2.py`, `analysis/figs2.py` |
| Fig. 5 (seeds), Supplementary Fig. S1 (training fractions) | `results/stats/contrasts.csv`, `analysis/figs2.py` |
| Table 5 (bottleneck audit), Supplementary Table S3 | `results/stats/audit.csv`, `analysis/tabs2.py` |
| Fig. 6 (qualitative examples) | `notebooks/04_qualitative.ipynb`, `analysis/fig_qual.py` (needs the prepared images, which are not redistributed) |
| Supplementary Table S4 (gap contrasts) | `results/stats/gap_contrasts.csv`, `analysis/gap.py` |
| Supplementary Table S5 (noise test) | `results/stats/noise_contrasts.csv`, `analysis/noise.py` |
| Supplementary Tables S6 to S10 | `analysis/tabs2.py` from `results/stats/` and `results/data/` |
| Supplementary Table S11 (equal factors) | `results/stats/equal_k_contrasts.csv`, `results/stats/equal_k_levels.csv`, `analysis/equal_k.py` |
| Normal images, costs, classes, sensitivity analysis | `normals_fp.csv`, `costs.csv`, `by_class_planA.csv`, `contrasts_no_annotated.csv` in `results/stats/` |

Running `src/analysis.py` on `results/data` reproduces every file in `results/stats/` exactly.

## Environment

Python 3.13, TensorFlow 2.21, Keras 3.13 and NumPy 2.4 on Google Colab with NVIDIA T4 GPUs. The statistics need only NumPy and pandas.

## Citation

See `CITATION.cff`. License: MIT (`LICENSE`).
