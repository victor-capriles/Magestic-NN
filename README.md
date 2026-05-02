# Magestic-NN

Magestic-NN is a master's thesis repository focused on reproducible activity classification for the HCV ChEMBL target `CHEMBL379`. The current workflow maintains parallel IC50 and EC50 tracks, builds curated binary classification datasets from raw bioactivity measurements, regenerates RDKit descriptors, applies leakage-safe preprocessing, and prepares the final scaled tabular inputs used by both the deep learning and classical ML notebooks.

## Repository Overview

- `datasets/CHEMBL379_IC50_AllDesc.csv`: raw tab-delimited IC50 ChEMBL export.
- `datasets/CHEMBL379_EC50_AllDesc.csv`: raw tab-delimited EC50 ChEMBL export.
- `datasets/postprocessed-CHEMBL379_IC50/`: IC50 preprocessing scripts and generated datasets.
- `datasets/postprocessed-CHEMBL379_EC50/`: EC50 preprocessing scripts and generated datasets.
- `deep_learning_pipeline.ipynb`: notebook for the main MLP experiments on the selected dataset track.
- `classical_ml_baseline_pipeline.ipynb`: notebook for the Logistic Regression, Random Forest, and SVM baseline sweep on the selected dataset track.

## Results At A Glance

The thesis primary selection metric is test F1. Test ROC-AUC is the main secondary metric, so the quick comparison below uses the best locked ML model and the best locked deep-learning model on each track under that policy.

| Track | Best classical ML | Best deep learning | Quick takeaway |
| --- | --- | --- | --- |
| IC50 | Random Forest baseline: test F1 = 0.8502, test ROC-AUC = 0.9398 | `small_mlp` from `deep_learning_pipeline.ipynb` experiment_001: test F1 = 0.8629 | Deep learning is currently better on IC50 because it wins on the thesis primary metric, test F1. Random Forest is still very competitive and slightly stronger on ROC-AUC and recall, so it remains the stronger ranking-oriented baseline rather than the best final thresholded classifier. |
| EC50 | Random Forest baseline: test F1 = 0.8638, test ROC-AUC = 0.9445 | `recommended_mlp` from `deep_learning_pipeline.ipynb` experiment_001: test F1 = 0.8629, test ROC-AUC = 0.9280 | Classical ML is currently better on EC50 because the Random Forest slightly beats the best MLP on test F1 and more clearly leads on ROC-AUC, accuracy, and precision. The MLP is still competitive and keeps a small recall edge, but not enough to lead the track overall. |

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

The commands below show the IC50 track explicitly. For the EC50 track, run the same stage sequence from `datasets/postprocessed-CHEMBL379_EC50/` instead of `datasets/postprocessed-CHEMBL379_IC50/`.

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

The EC50 track produces the same three files under `datasets/postprocessed-CHEMBL379_EC50/scaled_dataset/`.

Most preprocessing scripts also support optional arguments such as `--suffix`, `--output-dir`, and stage-specific overrides if you need to run comparison experiments without overwriting the default artifacts.

## Run The Notebook Pipeline

After the preprocessing steps finish, open either notebook depending on the comparison you want to run:

```powershell
code deep_learning_pipeline.ipynb
code classical_ml_baseline_pipeline.ipynb
```

Or launch Jupyter directly:

```powershell
jupyter notebook deep_learning_pipeline.ipynb
jupyter notebook classical_ml_baseline_pipeline.ipynb
```

Inside the notebook:

1. Select the Python kernel associated with your virtual environment.
2. Run the cells from top to bottom.
3. Set `DATASET_TRACK` in the first setup cell to `IC50` or `EC50` before running the experiment.
4. Confirm that the first cell detects the expected scaled dataset directory for the selected track.

The current notebook starts from the already scaled descriptor splits, so you do not need to rerun preprocessing unless you want to regenerate the artifacts or change preprocessing parameters.

