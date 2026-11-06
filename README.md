# Patient-level benchmark for lung cancer classification on IQ-OTH/NCCD

Deep models for lung cancer classification from CT scans routinely report accuracy
above 98% on the IQ-OTH/NCCD benchmark. The dataset contains 1,097 slices from only
110 patients, with 8 to 14 near-identical slices per patient, so a random slice-level
split places the same anatomy on both sides of the train/test divide.

This repository contains the artefacts needed to reproduce a **patient-level**
evaluation of that benchmark.

## Headline results

| Protocol | EfficientNetB0 | Notes |
|---|---|---|
| Slice-level split (as reported in the literature) | up to 99.5% | same patient on both sides |
| Patient-level, stratified group 5-fold CV | **87.84%** (AUC 0.9441) | this work |

The same three architectures (ResNet50, EfficientNetB0, ConvNeXt-Tiny) gain 0.6 to 4.1
percentage points when the split ignores patients, and none of them approach the
98-99.5% previously reported.

## Contents

| Path | Description |
|---|---|
| `data/dataset_manifest.csv` | one row per slice: relative path, **patient id**, class label, slice index |
| `code/run_v3.py` | training and evaluation pipeline (patient-level group CV) |
| `code/validate_ssim.py` | SSIM analysis of intra- vs inter-patient slice similarity |
| `code/gradcam_proof.py` | Grad-CAM visualisation of leakage under slice-level splitting |
| `results/all_results.json` | raw metrics for every model and fold |
| `requirements.txt` | Python dependencies |

## Data

The images are **not** redistributed here. Download the public dataset from the
original repository:

- IQ-OTH/NCCD, Mendeley Data, DOI: [10.17632/bhmdr45bh2.1](https://data.mendeley.com/datasets/bhmdr45bh2/1)

Place the extracted class folders under `data/raw/` so that paths in
`dataset_manifest.csv` resolve.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python code/validate_ssim.py      # slice similarity / leakage check
python code/run_v3.py             # patient-level 5-fold cross-validation
python code/gradcam_proof.py      # Grad-CAM evidence
```

All experiments use a fixed seed and the fold definitions derived from
`data/dataset_manifest.csv`, so repeated runs reproduce the reported numbers.

## How to cite

Lafta, A. L. (2026). *A Patient-Level Benchmark and Evaluation Protocol for Lung Cancer
Classification on the IQ-OTH/NCCD Dataset.* Manuscript submitted for publication.

## License

Code released under the MIT License. The dataset keeps its original license from
Mendeley Data.
