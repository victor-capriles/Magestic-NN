# Magestic-NN

Magestic-NN is a master's thesis repository focused on reproducible activity classification for the HCV ChEMBL target `CHEMBL379`. The current workflow builds a curated binary classification dataset from raw IC50 measurements, regenerates RDKit descriptors, applies leakage-safe preprocessing, and prepares the final scaled tabular inputs used by the deep learning notebook.

## Repository Overview

- `datasets/CHEMBL379_IC50_AllDesc.csv`: raw tab-delimited ChEMBL export.
- `datasets/postprocessed-CHEMBL379_IC50/`: preprocessing scripts and generated datasets.
- `deep_learning_pipeline.ipynb`: notebook that loads the final scaled split datasets for deep learning experiments.

## Prerequisites

- A recent Python 3 interpreter compatible with the pinned packages in `requirements.txt`.
- `pip` for dependency installation.
- Optional: an NVIDIA GPU for PyTorch acceleration. The notebook also runs on CPU.

## Environment Setup

The commands below assume you are running from the repository root.

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU acceleration on Windows, install the CUDA-enabled PyTorch wheel after the base requirements:

```powershell
python -m pip install --force-reinstall --no-deps torch==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### Bash

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Preprocessing Pipeline

If you want to reproduce the data pipeline from the raw export, run the scripts below in order.

```powershell
python datasets/postprocessed-CHEMBL379_IC50/build_classification_dataset.py
python datasets/postprocessed-CHEMBL379_IC50/build_rdkit_descriptor_dataset.py
python datasets/postprocessed-CHEMBL379_IC50/build_stratified_split_dataset.py
python datasets/postprocessed-CHEMBL379_IC50/build_variance_filtered_dataset.py
python datasets/postprocessed-CHEMBL379_IC50/build_correlation_filtered_dataset.py
python datasets/postprocessed-CHEMBL379_IC50/build_scaled_dataset.py
```

### What Each Step Produces

- `build_classification_dataset.py`: creates the clean binary classification dataset from the raw CHEMBL379 IC50 export and writes audit files for ambiguous or duplicate rows.
- `build_rdkit_descriptor_dataset.py`: regenerates RDKit descriptors directly from SMILES and records invalid SMILES or canonical-collision audits.
- `build_stratified_split_dataset.py`: builds reproducible train, validation, and test splits.
- `build_variance_filtered_dataset.py`: removes zero-variance descriptors using the training split only.
- `build_correlation_filtered_dataset.py`: removes highly correlated descriptors using the training split only.
- `build_scaled_dataset.py`: applies z-score scaling using training-set statistics only.

The final notebook-ready artifacts are written to:

- `datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/train_dataset.csv`
- `datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/validation_dataset.csv`
- `datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/test_dataset.csv`

Most preprocessing scripts also support optional arguments such as `--suffix`, `--output-dir`, and stage-specific overrides if you need to run comparison experiments without overwriting the default artifacts.

## Run The Notebook Pipeline

After the preprocessing steps finish, open the notebook:

```powershell
code deep_learning_pipeline.ipynb
```

Or launch Jupyter directly:

```powershell
jupyter notebook deep_learning_pipeline.ipynb
```

Inside the notebook:

1. Select the Python kernel associated with your virtual environment.
2. Run the cells from top to bottom.
3. Confirm that the first cell detects the expected dataset directory at `datasets/postprocessed-CHEMBL379_IC50/scaled_dataset`.

The current notebook starts from the already scaled descriptor splits, so you do not need to rerun preprocessing unless you want to regenerate the artifacts or change preprocessing parameters.

