# CATF-IDS

**Context-Adaptive Threat Fusion for Multi-Modal Intrusion Detection in Industrial IoT**

Code for the paper by **Abdlrahman Amr** and **Eman E. Sanad**
(Faculty of Computing and Artificial Intelligence, Cairo University).
Manuscript under review; the citation will be added on publication.

CATF-IDS scores network traffic, device telemetry and Linux host activity with a
separate random forest for each, fuses the three scores with an interpretable
logistic regression (an AUC-weighted ensemble of the five cross-validation fold
models), and decides through an adaptive layer whose thresholds, trust and
policy limits move only within bounds a site sets. It is evaluated on
[TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets).

## Results reported in the paper

| | Cross-validation (5 folds, balanced 13,256 records) | Full stream (24,934 records, time order) |
|---|---|---|
| F1 | 0.9050 ± 0.0019 | 0.9507 |
| Precision | 0.8499 | 0.9377 |
| Recall | 0.9679 | 0.9641 |
| ROC-AUC | 0.9750 | 0.9772 |
| False-alarm rate | 0.1709 | 0.1768 |

Agreement between the network and host labels after the timestamp join is
Cohen's κ = 0.126 at the pipeline's 300 s tolerance.

## Repository layout

```
code/
  catf_ids.py               the system (final configuration D by default)
  final_run.py              configuration C vs D, and removing each change in turn
  threshold_stress.py       adaptive vs frozen thresholds under different attack orderings
  policy_study.py           site-calibrated policy limits and injected device-reading jumps
  kappa_sweep.py            label agreement after the join at nine tolerances (1 s to 1 h)
  export_processed_data.py  optional: saves the fused dataset the paper evaluates
figures/                    scripts that draw the paper's figures (vector PDF/SVG + PNG)
data/
  README.md                 how to obtain TON_IoT and where to put it
```

## Quick start

1. **Get the code.** Either clone it:
   ```
   git clone https://github.com/amra88189-arch/catf-ids-paper.git
   cd catf-ids-paper
   ```
   or click **Code → Download ZIP** on this page and unzip it.

2. **Install the requirements** (Python 3.10 or later):
   ```
   pip install -r requirements.txt
   ```

3. **Check that it runs** (synthetic data, no download needed, under a minute):
   ```
   cd code
   python catf_ids.py --test
   ```

4. **Get the dataset.** The data is not stored in this repository. Download
   TON_IoT from UNSW Canberra (free for academic research):
   **https://research.unsw.edu.au/projects/toniot-datasets**
   From the download, take the **Processed_datasets** folder (it holds
   `Processed_Network_dataset`, `Processed_IoT_dataset`, `Processed_Linux_dataset`
   and `Processed_Windows_dataset`). The 38 files used are listed in
   [data/README.md](data/README.md).

5. **Run the paper's configuration** from the `code/` folder, pointing `--base` at
   that folder:
   ```
   python catf_ids.py --base "path/to/Processed_datasets"
   ```
   On Windows, for example: `python catf_ids.py --base "D:\TON_IoT\Processed_datasets"`.
   Instead of `--base` you can place the folder at `data/TON_IoT/Processed_datasets/`
   or set the `CATF_DATA` environment variable.

The sections below list every script and the result in the paper it produces.

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

`requirements.txt` gives minimum versions. Random-forest results can differ
slightly between scikit-learn versions; `export_processed_data.py` records the
exact versions of a run in its `manifest.json`.

## Data

Download the TON_IoT **Processed_datasets** folder from UNSW Canberra and place it at
`data/TON_IoT/Processed_datasets/`, or pass its location with `--base` (every
script accepts it) or the `CATF_DATA` environment variable. The expected files are
listed in [data/README.md](data/README.md).

The processed, fused dataset the paper evaluates is not stored here because of its
size. `python export_processed_data.py --base "path/to/Processed_datasets"` rebuilds
it from the download in a few minutes (24,934 fused records and the balanced
13,256-record training set, with a manifest of counts, checksums and versions) and
checks the counts against the paper.

## Running

Run everything from the `code/` folder.

**Quick check, no dataset needed** (synthetic data, under a minute):

```
python catf_ids.py --test
```

**The paper's main results** (configuration D: cross-validation, full stream,
per-class detection, confusion matrices, records unseen by the layer models):

```
python catf_ids.py --base "<folder with the Processed_*_dataset folders>"
```

With no switches the script runs the final configuration. `--config_c` reproduces
the earlier configuration C. Models and plots are written to `models/` and `plots/`.

**Paper results by script**

| Paper | Command |
|---|---|
| Main results, per-class detection, Fig. 5 counts | `python catf_ids.py` |
| Removing each change in turn (attribution) | `python final_run.py` |
| Adaptive vs frozen thresholds; Fig. 6 trace data | `python threshold_stress.py` |
| Policy limits and injected device-reading jumps (Section 6.6, Fig. 7) | `python policy_study.py` |
| Label-free adaptation (Section 6.7) | `python catf_ids.py --p2_recal`<br>`python catf_ids.py --p2_recal --p2_pseudo`<br>`python catf_ids.py --p2_recal --p2_replay_only` |
| Cross-modal alignment (Section 5, Fig. 4) | `python kappa_sweep.py` |
| Processed dataset (optional) | `python export_processed_data.py` |

Add `--base "<folder>"` to any of them if the dataset is not in the default place.
`final_run.py` prepares the data once and runs all configurations (about 15 minutes).
`final_run.py`, `threshold_stress.py` and `policy_study.py` check their results against the paper's numbers and report any difference.

**Figures** (from the `figures/` folder; `pip install cairosvg` for the diagram
PDFs/PNGs):

| Figure | Script |
|---|---|
| Fig. 1 — IIoT tiers and Purdue levels | `make_fig_iiot_tiers.py` |
| Fig. 2 — TON_IoT testbed and the streams used | `make_fig_testbed.py` |
| Fig. 3 — the nine stages | `make_fig_nine_stages.py` |
| Fig. 4 — alignment sweep | `make_fig_kappa.py` |
| Figs. 5 and 7 — confusion matrices, injected jumps | `make_fig_results_D.py` |
| Fig. 6 — threshold trace | `make_fig_threshold_trace.py` (reads `threshold_stress.py` output) |

The numbers in `make_fig_kappa.py` and `make_fig_results_D.py` are copied from the
run outputs named in each script's header.

## Notes on the evaluation

- The three streams are joined on the nearest timestamp within 300 s; there is no
  shared key, and one device or host reading can be joined to several network
  records. Denial-of-service records have no device reading within 300 s and are
  removed by the join.
- No raw address or port field is a model input, but seven network features are
  aggregated per source address, an eighth combines two of them, and the
  destination port shapes one further feature and four signature scores.
- The full stream is partly in-sample: the fusion ensemble has seen every normal
  record. The paper also reports the records unseen by the layer models.

## Data licence and citation

TON_IoT is © UNSW Canberra. Free use for academic research is granted by its
authors; commercial use requires permission from Dr Nour Moustafa. If you use the
data, cite the TON_IoT papers listed in [data/README.md](data/README.md).

## Contact

Abdlrahman Amr — amra88189@gmail.com
