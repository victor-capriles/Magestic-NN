# Dataset Snapshot

Date: 2026-05-01
Source file: datasets/CHEMBL379_IC50_AllDesc.csv

- Total rows: 2111
- Total columns: 126
- Metadata and assay columns: 7
- Descriptor columns: 119
- All rows correspond to IC50 measurements in nM for target CHEMBL379
- No missing values were found in the core fields: Molecule ChEMBL ID, Smiles, Standard Type, Standard Relation, Standard Value, Standard Units, Target ChEMBL ID
- Unique Molecule ChEMBL IDs: 2111
- Unique SMILES strings: 2111
- Exact duplicate SMILES across records: 0
- Standard Relation distribution: '=' = 1840, '>' = 231, '<' = 40
- The dataset is tab-delimited and must be loaded accordingly

# Assay Semantics

- Standard Relation '=' means the IC50 value is exact
- Standard Relation '>' means the true IC50 is greater than the reported value
- Standard Relation '<' means the true IC50 is lower than the reported value
- Lower IC50 implies stronger inhibition, so relation symbols must be interpreted before assigning binary activity labels

# Verified Preprocessing Facts

- Zero-variance descriptor columns detected and marked for removal:
	- slogp_VSA9
	- smr_VSA8
	- MQN18
	- MQN22
	- MQN23
- These columns are constant across all molecules and do not provide predictive information

# Accepted Preprocessing Decisions

- Remove the five zero-variance descriptor columns before modeling:
	- slogp_VSA9
	- smr_VSA8
	- MQN18
	- MQN22
	- MQN23
- Reason: they are constant across all rows, contain no predictive signal, and only add unnecessary dimensionality
- In the implemented pipeline, zero-variance filtering should still be fit on the training split only to avoid leakage, even though these five descriptors are already verified as constant in the current dataset snapshot

# Labeling Policy v1

- The binary activity label will be derived from IC50 values using a threshold computed from exact measurements only
- Censored values with relation '>' and '<' are excluded from threshold estimation because they are bounds, not exact values
- Median IC50 using only exact rows ('=') is 310 nM
- This median-based threshold is very close to the tutor's suggested 300 nM cutoff and can be justified as the primary threshold for the classification task

# Threshold Computation

- The 310 nM threshold was computed using only the 1840 rows with Standard Relation '='
- Procedure used:
	- keep only exact IC50 rows
	- extract Standard Value in nM
	- sort the exact IC50 values in ascending order
	- because 1840 is an even number, the median is the average of the two central values in the sorted list
	- the central positions are 920 and 921 in 1-based indexing
	- both central values are 310 nM, so the median is 310 nM
- This makes 310 nM a reproducible data-driven threshold derived directly from the exact measurements in the dataset

If an alternative threshold is needed later, it should still be computed from exact rows only. Defensible options include:

- Quantile-based cutoff: choose another percentile from the exact IC50 distribution instead of the median
- pIC50-based cutoff: convert IC50 values to pIC50 and define the threshold on the transformed scale, then convert back if needed

Examples from the current exact-value distribution:

- Q1 = 31.62 nM
- Median = 310 nM
- Q3 = 4180 nM

General rule for any future threshold choice:

- estimate the threshold from exact rows only
- document the reason for that choice before training
- apply the same censor-aware labeling rules for '=', '>', and '<'
- do not choose the threshold based on which one gives the best model accuracy, because that would leak outcome information into dataset design

Definition:

- activity = 1 if IC50 <= 310 nM
- activity = 0 if IC50 > 310 nM

Row-level labeling rules:

- If Standard Relation is '=': assign the label directly from Standard Value
- If Standard Relation is '>':
	- assign activity = 0 when the reported bound is >= 310 nM
	- exclude the row if the reported bound is < 310 nM because the true IC50 remains ambiguous with respect to the threshold
- If Standard Relation is '<':
	- assign activity = 1 when the reported bound is <= 310 nM
	- exclude the row if the reported bound is > 310 nM because the true IC50 remains ambiguous with respect to the threshold

Counts under this policy:

- Exact rows labeled active: 923
- Exact rows labeled inactive: 917
- '>' rows safely labeled inactive: 231
- '<' rows safely labeled active: 37
- Ambiguous censored rows excluded: 3
- Total rows retained after labeling: 2108

# Duplicate SMILES Assessment

- A case-sensitive Python check on the raw export found 2111 unique SMILES strings across 2111 rows, so there are no exact duplicate SMILES in the current dataset snapshot
- As a result, no SMILES-level deduplication is needed for the current raw export before descriptor regeneration
- The labeled classification dataset currently retains 2108 rows after excluding the 3 ambiguous censored rows
- Active rows under the 310 nM threshold: 960
- Inactive rows under the 310 nM threshold: 1148

Implication for the pipeline:

- build the labeled classification dataset directly from the censor-aware labeling step
- if descriptor regeneration later produces canonical SMILES collisions, handle deduplication at that canonicalized stage rather than from the current raw SMILES strings

Open follow-up for regression:

- define the regression dataset from exact IC50 rows only, then revisit deduplication only if canonicalized structures collide after descriptor regeneration

# Classification Dataset Builder

- Script created: datasets/postprocessed-CHEMBL379_IC50/build_classification_dataset.py
- Shared helper module created: datasets/postprocessed-CHEMBL379_IC50/common.py
- Purpose: build the clean base dataset for binary classification before feature filtering, scaling, and DNN training
- Default output folder: datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset/
- The builder now accepts optional `--output-dir` and `--suffix` arguments so comparison runs can write to a separate folder without overwriting prior artifacts
- The script performs these steps:
	- read the raw tab-delimited CHEMBL379 IC50 export
	- keep only rows within thesis scope: IC50, nM, CHEMBL379
	- compute the 310 nM threshold from exact rows only
	- assign the binary activity label using the censor-aware policy
	- exclude ambiguous censored rows
	- group rows by case-sensitive SMILES to detect exact duplicates
	- keep one representative row only if an exact duplicate group is label-concordant
	- exclude a group only if an exact duplicate group is label-conflicting
- In the current dataset snapshot, the duplicate-handling branch did not alter the data because no exact duplicate SMILES were found

# Generated Classification Outputs

- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset/classification_base_dataset.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset/excluded_ambiguous_censored.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset/collapsed_concordant_duplicates.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset/excluded_conflicting_duplicates.csv
- Summary file created: datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset/classification_curation_summary.json

Current generated counts:

- input rows: 2111
- threshold: 310.0 nM
- labeled rows after censor-aware filtering: 2108
- ambiguous censored rows excluded: 3
- collapsed duplicate groups: 0
- conflicting duplicate groups: 0
- final classification rows: 2108
- final active rows: 960
- final inactive rows: 1148
- Safe validation rerun also completed successfully into datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset_validation/ with the same counts, confirming that `--suffix validation` avoids overwriting the default artifacts

# RDKit Descriptor Regeneration

- Script created: datasets/postprocessed-CHEMBL379_IC50/build_rdkit_descriptor_dataset.py
- Purpose: regenerate the descriptor matrix directly from SMILES for the clean classification dataset instead of relying on the original exported descriptor values
- Default classification input: datasets/postprocessed-CHEMBL379_IC50/classification_curation_dataset/classification_base_dataset.csv
- The builder now accepts optional `--classification-dataset`, `--output-dir`, and `--suffix` arguments so alternate curation runs can feed matching RDKit outputs without replacing existing artifacts
- Implementation note: descriptor computation is performed on hydrogen-added RDKit molecules because this matches the original export convention for columns such as LabuteASA, Chi0v, kappa1, VSA descriptors, and NumAtoms
- Implementation note: NumRotatableBonds uses RDKit's non-strict option because this matches the original export convention
- The regenerated dataset keeps the classification metadata and appends the 119 descriptor columns in the same order as the original export
- Canonical SMILES are also generated to audit whether deduplication becomes necessary after canonicalization

# Generated RDKit Outputs

- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/rdkit_descriptor_dataset/classification_rdkit_descriptor_dataset.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/rdkit_descriptor_dataset/rdkit_invalid_smiles.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/rdkit_descriptor_dataset/rdkit_canonical_smiles_collisions.csv
- Summary file created: datasets/postprocessed-CHEMBL379_IC50/rdkit_descriptor_dataset/rdkit_descriptor_summary.json

Current regenerated descriptor counts:

- input classification rows: 2108
- descriptor rows written: 2108
- invalid SMILES rows: 0
- canonical collision groups: 0
- descriptor column count: 119
- Safe validation rerun also completed successfully into datasets/postprocessed-CHEMBL379_IC50/rdkit_descriptor_dataset_validation/ using the suffixed classification dataset as input, confirming that alternate curation runs can feed matching RDKit outputs without replacing the default artifacts

# Modeling Split

- Script created: datasets/postprocessed-CHEMBL379_IC50/build_stratified_split_dataset.py
- Shared preprocessing helper module created: datasets/postprocessed-CHEMBL379_IC50/preprocessing_common.py
- Purpose: create the fixed train/validation/test split that later variance filtering, correlation filtering, scaling, and model fitting must learn from without touching held-out rows
- Default input dataset: datasets/postprocessed-CHEMBL379_IC50/rdkit_descriptor_dataset/classification_rdkit_descriptor_dataset.csv
- Default output folder: datasets/postprocessed-CHEMBL379_IC50/stratified_split_dataset/
- The builder accepts optional `--input-dataset`, `--output-dir`, and `--suffix` arguments, plus configurable split fractions and random seed
- Current default split policy:
	- stratified by binary activity label
	- train fraction = 0.8
	- validation fraction = 0.1
	- test fraction = 0.1
	- random seed = 42

# Generated Split Outputs

- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/stratified_split_dataset/train_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/stratified_split_dataset/validation_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/stratified_split_dataset/test_dataset.csv
- Summary file created: datasets/postprocessed-CHEMBL379_IC50/stratified_split_dataset/split_summary.json

Current split counts:

- total rows assigned: 2108
- train rows: 1686
- validation rows: 211
- test rows: 211
- train active/inactive: 768 / 918
- validation active/inactive: 96 / 115
- test active/inactive: 96 / 115

# Variance Filtering

- Script created: datasets/postprocessed-CHEMBL379_IC50/build_variance_filtered_dataset.py
- Purpose: fit descriptor variance filtering on the training split only and apply the same kept-column set to validation and test without leakage
- Default input folder: datasets/postprocessed-CHEMBL379_IC50/stratified_split_dataset/
- Default output folder: datasets/postprocessed-CHEMBL379_IC50/variance_filtered_dataset/
- The builder accepts optional `--input-dir`, `--output-dir`, and `--suffix` arguments, plus a configurable `--variance-threshold`
- Current default variance policy:
	- fit on train_dataset.csv only
	- variance threshold = 0.0
	- remove descriptors whose training variance is at or below the threshold

# Generated Variance Filter Outputs

- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/variance_filtered_dataset/train_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/variance_filtered_dataset/validation_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/variance_filtered_dataset/test_dataset.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/variance_filtered_dataset/dropped_descriptor_columns.csv
- Summary file created: datasets/postprocessed-CHEMBL379_IC50/variance_filtered_dataset/variance_filter_summary.json

Current variance filter result:

- input descriptor count: 119
- kept descriptor count: 114
- dropped descriptor count: 5
- dropped descriptors:
	- slogp_VSA9
	- smr_VSA8
	- MQN18
	- MQN22
	- MQN23
- train rows after filtering: 1686
- validation rows after filtering: 211
- test rows after filtering: 211

# Correlation Filtering

- Script created: datasets/postprocessed-CHEMBL379_IC50/build_correlation_filtered_dataset.py
- Purpose: fit descriptor correlation filtering on the training split only and apply the same kept-column set to validation and test without leakage
- Default input folder: datasets/postprocessed-CHEMBL379_IC50/variance_filtered_dataset/
- Default output folder: datasets/postprocessed-CHEMBL379_IC50/correlation_filtered_dataset/
- The builder accepts optional `--input-dir`, `--output-dir`, and `--suffix` arguments, plus a configurable `--correlation-threshold`
- Current default correlation policy:
	- fit on train_dataset.csv only
	- absolute Pearson threshold = 0.8
	- if a descriptor exceeds the threshold with an earlier kept descriptor, keep the earlier descriptor and drop the later one

# Generated Correlation Filter Outputs

- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/correlation_filtered_dataset/train_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/correlation_filtered_dataset/validation_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/correlation_filtered_dataset/test_dataset.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/correlation_filtered_dataset/dropped_correlated_descriptor_columns.csv
- Summary file created: datasets/postprocessed-CHEMBL379_IC50/correlation_filtered_dataset/correlation_filter_summary.json

Current correlation filter result:

- input descriptor count: 114
- kept descriptor count: 71
- dropped descriptor count: 43
- train rows after filtering: 1686
- validation rows after filtering: 211
- test rows after filtering: 211
- dropped descriptors:
	- LabuteASA
	- AMW
	- ExactMW
	- NumLipinskiHBA
	- NumLipinskiHBD
	- NumHBD
	- NumHBA
	- NumHeteroAtoms
	- NumHeavyAtoms
	- NumAtoms
	- NumAliphaticRings
	- NumSaturatedCarbocycles
	- NumAliphaticCarbocycles
	- Chi0v
	- Chi1v
	- Chi2v
	- Chi3v
	- Chi1n
	- Chi2n
	- Chi3n
	- Chi4n
	- kappa2
	- kappa3
	- slogp_VSA5
	- slogp_VSA6
	- smr_VSA1
	- smr_VSA5
	- smr_VSA7
	- MQN1
	- MQN10
	- MQN12
	- MQN13
	- MQN14
	- MQN16
	- MQN17
	- MQN19
	- MQN20
	- MQN21
	- MQN25
	- MQN28
	- MQN31
	- MQN32
	- MQN42
- Example retained-pair decisions from the audit file:
	- LabuteASA dropped against SMR with |r| = 0.9864
	- NumLipinskiHBA dropped against TPSA with |r| = 0.9666
	- kappa3 dropped against NumRotatableBonds with |r| = 0.9253

# Scaling

- Script created: datasets/postprocessed-CHEMBL379_IC50/build_scaled_dataset.py
- Purpose: fit descriptor scaling on the training split only after correlation filtering and apply the same descriptor-wise statistics to validation and test
- Default input folder: datasets/postprocessed-CHEMBL379_IC50/correlation_filtered_dataset/
- Default output folder: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/
- Current default scaling policy:
	- z-score standardization
	- fit descriptor means and standard deviations on train_dataset.csv only
	- preserve metadata columns and the binary activity label unchanged

# Generated Scaling Outputs

- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/train_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/validation_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/test_dataset.csv
- Audit file created: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/scaling_parameters.csv
- Summary file created: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/scaling_summary.json

Current scaling result:

- scaled descriptor count: 71
- train rows after scaling: 1686
- validation rows after scaling: 211
- test rows after scaling: 211
- numerical validation on the scaled training split:
	- maximum absolute descriptor mean: 4.5907e-16
	- maximum absolute deviation from unit standard deviation: 1.1102e-16

# Deep Learning Notebook Wiring

- Notebook in use: deep_learning_pipeline.ipynb
- The notebook now includes a thirteenth results-export cell, and the latest saved notebook outputs include completed runs for all three MLP experiments plus the exported comparison summary
- The notebook has now been upgraded again so future runs create a numbered artifact folder under results/experiment_###/ before training starts
- The setup cell now resolves the repository root dynamically by scanning upward from the current working directory, so the notebook no longer depends on a machine-specific absolute path
- Current implemented notebook stages:
	- setup cell
	- run-directory setup cell
	- data loading cell
	- sanity-check cell
	- feature and target preparation cell
	- DataLoader cell
	- model-definition cell
	- shared training and evaluation utilities cell
	- small MLP experiment cell
	- recommended MLP experiment cell
	- large MLP experiment cell
	- confusion-matrix comparison cell
	- results export cell
- Current notebook data source: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/
- Validated notebook state after execution:
	- loaded split shapes: train = (1686, 82), validation = (211, 82), test = (211, 82)
	- descriptor column count after preprocessing: 71
	- tensor shapes: X_train = (1686, 71), X_validation = (211, 71), X_test = (211, 71)
	- tensor shapes: y_train = (1686, 1), y_validation = (211, 1), y_test = (211, 1)
	- DataLoader state: train batches = 53, validation batches = 4, test batches = 4
	- sample training batch shape: X = (32, 71), y = (32, 1)
	- architecture candidates currently defined in the notebook:
		- small_mlp = 71 -> 32 -> 1, 2337 trainable parameters
		- recommended_mlp = 71 -> 64 -> 32 -> 1, 6721 trainable parameters
		- large_mlp = 71 -> 128 -> 64 -> 32 -> 1, 19585 trainable parameters
	- the recommended baseline model is instantiated in the notebook with dropout = 0.2
	- shared training defaults currently defined in the notebook:
		- max epochs = 100
		- optimizer = AdamW
		- learning rate = 0.001
		- weight decay = 0.0001
		- loss = BCEWithLogitsLoss
		- training runs the full epoch budget and restores the checkpoint with the minimum validation loss at the end
	- desktop comparison run executed and exported with these run metadata:
		- run timestamp = 2026-05-02_01-56-27
		- device = cuda
		- torch version = 2.11.0+cu130
		- GPU = NVIDIA GeForce RTX 5080
	- exported comparison artifacts created:
		- results/deep_learning_results_2026-05-02_01-56-27.csv
		- results/deep_learning_results_2026-05-02_01-56-27.json
	- small MLP results, architecture = 71 -> 32 -> 1:
		- epochs trained = 44
		- validation loss = 0.3754
		- validation accuracy = 0.8389
		- validation F1 = 0.8247
		- validation ROC-AUC = 0.9169
		- test loss = 0.3191
		- test accuracy = 0.8626
		- test F1 = 0.8557
		- test ROC-AUC = 0.9387
	- recommended MLP results, architecture = 71 -> 64 -> 32 -> 1:
		- epochs trained = 37
		- validation loss = 0.3636
		- validation accuracy = 0.8626
		- validation F1 = 0.8543
		- validation ROC-AUC = 0.9274
		- test loss = 0.3360
		- test accuracy = 0.8720
		- test F1 = 0.8670
		- test ROC-AUC = 0.9300
	- large MLP results, architecture = 71 -> 128 -> 64 -> 32 -> 1:
		- epochs trained = 41
		- validation loss = 0.3557
		- validation accuracy = 0.8578
		- validation F1 = 0.8421
		- validation ROC-AUC = 0.9336
		- test loss = 0.3797
		- test accuracy = 0.8389
		- test F1 = 0.8317
		- test ROC-AUC = 0.9259
	- desktop exploratory-run takeaway from that run:
		- recommended_mlp was the strongest model in that earlier exploratory run on the held-out test set by accuracy and F1
		- small_mlp achieved the highest test ROC-AUC of the three models
		- large_mlp did not improve held-out generalization on this split despite its larger capacity
	- notebook usability improvements now implemented for future runs:
		- epoch-level progress bars during model training
		- live training dashboard showing train-loss vs validation-loss curves and validation metrics while an experiment is running
		- test-set confusion-matrix comparison cell for the available experiment results
		- numbered run-directory creation under results/experiment_###/
		- the run-directory setup cell now reuses the current RUN_DIR when rerun in the same kernel session, so repeated execution does not accidentally advance to a new experiment folder
		- training now always uses the full epoch budget instead of stopping early; the notebook still restores the checkpoint from the epoch with the best validation loss at the end of training
		- experiment summaries and exported result files now include best_epoch so the selected checkpoint can be distinguished from the full number of completed epochs
		- export cell now saves a per-run summary CSV and JSON, per-model history CSVs, per-model training-curve PNGs, and a combined confusion-matrix PNG
	- laptop-side validation after these notebook edits:
		- setup through the shared training-utilities cell executed successfully on the laptop kernel
		- the numbered run-directory setup resolved the next run folder as results/experiment_002 because the earlier desktop run already created results/experiment_001-style artifacts
		- rerunning the run-directory setup cell in the same kernel session reused results/experiment_002 instead of allocating results/experiment_003
		- a smoke-check experiment with max_epochs = 3 completed all 3 epochs and reported best_epoch correctly, confirming the full-budget checkpoint-selection behavior
		- end-to-end validation of the updated confusion-matrix and export cells still requires rerunning the main experiment cells after the notebook upgrade
	- official first full numbered run executed and exported with these run metadata:
		- run directory = results/experiment_001
		- run timestamp = 2026-05-02T15:05:03
		- device = cpu
		- torch version = 2.11.0+cpu
		- CUDA available = false
	- exported official-run artifacts created:
		- results/experiment_001/summary.csv
		- results/experiment_001/summary.json
		- results/experiment_001/test_confusion_summary.csv
		- results/experiment_001/history/*.csv
		- results/experiment_001/plots/*.png
	- official run results, architecture = 71 -> 32 -> 1 for small_mlp:
		- epochs trained = 100
		- best epoch = 39
		- validation loss = 0.3777
		- validation accuracy = 0.8389
		- validation F1 = 0.8247
		- validation ROC-AUC = 0.9179
		- test loss = 0.3279
		- test accuracy = 0.8720
		- test precision = 0.8416
		- test recall = 0.8854
		- test F1 = 0.8629
		- test ROC-AUC = 0.9341
		- test confusion matrix = TN 99, FP 16, FN 11, TP 85
	- official run results, architecture = 71 -> 64 -> 32 -> 1 for recommended_mlp:
		- epochs trained = 100
		- best epoch = 13
		- validation loss = 0.3770
		- validation accuracy = 0.8578
		- validation F1 = 0.8454
		- validation ROC-AUC = 0.9157
		- test loss = 0.3200
		- test accuracy = 0.8531
		- test precision = 0.8283
		- test recall = 0.8542
		- test F1 = 0.8410
		- test ROC-AUC = 0.9349
		- test confusion matrix = TN 98, FP 17, FN 14, TP 82
	- official run results, architecture = 71 -> 128 -> 64 -> 32 -> 1 for large_mlp:
		- epochs trained = 100
		- best epoch = 14
		- validation loss = 0.3627
		- validation accuracy = 0.8578
		- validation F1 = 0.8421
		- validation ROC-AUC = 0.9245
		- test loss = 0.3368
		- test accuracy = 0.8436
		- test precision = 0.8058
		- test recall = 0.8646
		- test F1 = 0.8342
		- test ROC-AUC = 0.9297
		- test confusion matrix = TN 95, FP 20, FN 13, TP 83
	- official-run training-curve interpretation:
		- all three models completed the full 100 epochs, so the current no-early-stop plus best-checkpoint policy behaved exactly as intended
		- small_mlp showed the mildest overfitting: validation loss bottomed at epoch 39 and then rose from 0.3777 to 0.4517 by epoch 100 while validation F1 remained fairly stable around 0.82 to 0.83
		- recommended_mlp overfit much earlier: validation loss bottomed at epoch 13 and then rose to 0.5938 by epoch 100, which suggests the deeper network became increasingly overconfident even though validation F1 and ROC-AUC stayed broadly stable
		- large_mlp overfit the most aggressively: validation loss bottomed at epoch 14 and then rose to 0.7466 by epoch 100, so the extra capacity clearly exceeded what this split can support
		- for recommended_mlp and large_mlp, checkpoint restoration is essential because the final-epoch model would be much worse calibrated than the selected best-validation-loss checkpoint
	- official-run comparison takeaway:
		- small_mlp is the strongest model on the held-out test set by thresholded classification behavior: it achieved the best accuracy, precision, recall, F1, balanced error profile, and the fewest total mistakes at 27 out of 211 test samples
		- recommended_mlp achieved the lowest test loss and the highest test ROC-AUC, but both margins over small_mlp are tiny, so the two smaller models are effectively very close on ranking-based metrics
		- large_mlp again failed to convert its stronger validation-side loss and ROC-AUC into the best held-out test performance, which reinforces the conclusion that higher capacity is not helping on the current dataset and split
		- across the exploratory desktop run and the baseline official experiment_001 run, the large model remained the weakest practical option on held-out test accuracy and F1 before the later tuned experiment_002 rerun
		- under the baseline hyperparameter setting, the lead between small_mlp and recommended_mlp is not fully stable across runs, so the thesis should define the primary model-selection metric before naming one architecture as the single best baseline

# Tuned Hyperparameter Run: experiment_002

- Tuned run executed and exported with these run metadata:
	- run directory = results/experiment_002
	- run timestamp = 2026-05-02T16:52:32
	- device = cpu
	- torch version = 2.11.0+cpu
	- CUDA available = false
- Global notebook defaults used for this run:
	- dropout rate = 0.3
	- learning rate = 1e-4
	- weight decay = 5e-4
	- max epochs = 100
- Exported tuned-run artifacts created:
	- results/experiment_002/summary.csv
	- results/experiment_002/summary.json
	- results/experiment_002/test_confusion_summary.csv
	- results/experiment_002/history/*.csv
	- results/experiment_002/plots/*.png
- Tuned run results, architecture = 71 -> 32 -> 1 for small_mlp:
	- epochs trained = 100
	- best epoch = 100
	- validation loss = 0.4137
	- validation accuracy = 0.8199
	- validation F1 = 0.8000
	- validation ROC-AUC = 0.8934
	- test loss = 0.3588
	- test accuracy = 0.8389
	- test precision = 0.8298
	- test recall = 0.8125
	- test F1 = 0.8211
	- test ROC-AUC = 0.9241
	- test confusion matrix = TN 99, FP 16, FN 18, TP 78
- Tuned run results, architecture = 71 -> 64 -> 32 -> 1 for recommended_mlp:
	- epochs trained = 100
	- best epoch = 99
	- validation loss = 0.3959
	- validation accuracy = 0.8531
	- validation F1 = 0.8360
	- validation ROC-AUC = 0.9083
	- test loss = 0.3369
	- test accuracy = 0.8341
	- test precision = 0.8211
	- test recall = 0.8125
	- test F1 = 0.8168
	- test ROC-AUC = 0.9290
	- test confusion matrix = TN 98, FP 17, FN 18, TP 78
- Tuned run results, architecture = 71 -> 128 -> 64 -> 32 -> 1 for large_mlp:
	- epochs trained = 100
	- best epoch = 65
	- validation loss = 0.3869
	- validation accuracy = 0.8531
	- validation F1 = 0.8410
	- validation ROC-AUC = 0.9111
	- test loss = 0.3286
	- test accuracy = 0.8626
	- test precision = 0.8317
	- test recall = 0.8750
	- test F1 = 0.8528
	- test ROC-AUC = 0.9322
	- test confusion matrix = TN 98, FP 17, FN 12, TP 84
- Tuned-run training-curve interpretation:
	- the stronger regularization plus lower learning rate substantially changed the optimization regime compared with experiment_001
	- small_mlp no longer showed the earlier overfitting pattern: validation loss kept decreasing almost monotonically through epoch 100 and the best checkpoint was the final epoch, which indicates the run became much more conservative and likely underfit relative to the baseline setting
	- recommended_mlp also shifted from early overfitting to near-monotonic improvement through the end of training; however, the stabilized training did not translate into better held-out test performance than experiment_001
	- large_mlp benefited the most from the tuned regime: the best epoch moved from 14 in experiment_001 to 65 here, the validation-loss curve flattened instead of diverging sharply, and the late-epoch degradation became mild rather than severe
	- overall, the tuned regime appears to have reduced overfitting successfully, but for the smaller models it likely traded too much learning capacity or optimization speed for regularization
- Direct comparison versus experiment_001:
	- the model ranking changed: large_mlp became the best model on the held-out test set by accuracy, precision, recall, F1, ROC-AUC, and test loss under the tuned hyperparameter regime
	- small_mlp regressed noticeably versus experiment_001: test F1 dropped from 0.8629 to 0.8211 and false negatives increased from 11 to 18 while false positives stayed at 16
	- recommended_mlp also regressed versus experiment_001: test F1 dropped from 0.8410 to 0.8168 and false negatives increased from 14 to 18 while false positives stayed at 17
	- large_mlp improved materially versus experiment_001: test accuracy rose from 0.8436 to 0.8626, test F1 rose from 0.8342 to 0.8528, ROC-AUC rose from 0.9297 to 0.9322, and the confusion matrix improved from TN 95 / FP 20 / FN 13 / TP 83 to TN 98 / FP 17 / FN 12 / TP 84
	- these results suggest that the lower learning rate and stronger regularization are helpful for the higher-capacity model, but too conservative for the small and recommended models under the current 100-epoch budget
- Tuned-run comparison takeaway:
	- experiment_002 does not replace experiment_001; it establishes a second, more regularized comparison point with a different optimization regime
	- under the tuned regime, large_mlp is the strongest current candidate on the held-out test set and is no longer the weakest option
	- architecture conclusions are therefore hyperparameter-sensitive in this project; the apparent winner can change when regularization and learning-rate settings change materially
	- a later experiment_003 follow-up run then tested the middle learning rate of 5e-4 with a 150-epoch budget and recovered performance for all three models relative to experiment_002 while keeping large_mlp strongest

# Follow-up Tuned Run: experiment_003

- Follow-up tuned run executed and exported with these run metadata:
	- run directory = results/experiment_003
	- run timestamp = 2026-05-02T17:15:03
	- device = cpu
	- torch version = 2.11.0+cpu
	- CUDA available = false
- Global notebook defaults used for this run:
	- dropout rate = 0.3
	- learning rate = 5e-4
	- weight decay = 5e-4
	- max epochs = 150
- Exported follow-up artifacts created:
	- results/experiment_003/summary.csv
	- results/experiment_003/summary.json
	- results/experiment_003/test_confusion_summary.csv
	- results/experiment_003/history/*.csv
	- results/experiment_003/plots/*.png
- Follow-up tuned run results, architecture = 71 -> 32 -> 1 for small_mlp:
	- epochs trained = 150
	- best epoch = 47
	- validation loss = 0.3883
	- validation accuracy = 0.8341
	- validation F1 = 0.8168
	- validation ROC-AUC = 0.9091
	- test loss = 0.3228
	- test accuracy = 0.8531
	- test precision = 0.8218
	- test recall = 0.8646
	- test F1 = 0.8426
	- test ROC-AUC = 0.9358
	- test confusion matrix = TN 97, FP 18, FN 13, TP 83
- Follow-up tuned run results, architecture = 71 -> 64 -> 32 -> 1 for recommended_mlp:
	- epochs trained = 150
	- best epoch = 19
	- validation loss = 0.3879
	- validation accuracy = 0.8341
	- validation F1 = 0.8168
	- validation ROC-AUC = 0.9098
	- test loss = 0.3336
	- test accuracy = 0.8483
	- test precision = 0.8265
	- test recall = 0.8438
	- test F1 = 0.8351
	- test ROC-AUC = 0.9330
	- test confusion matrix = TN 98, FP 17, FN 15, TP 81
- Follow-up tuned run results, architecture = 71 -> 128 -> 64 -> 32 -> 1 for large_mlp:
	- epochs trained = 150
	- best epoch = 17
	- validation loss = 0.3720
	- validation accuracy = 0.8720
	- validation F1 = 0.8615
	- validation ROC-AUC = 0.9159
	- test loss = 0.3256
	- test accuracy = 0.8673
	- test precision = 0.8269
	- test recall = 0.8958
	- test F1 = 0.8600
	- test ROC-AUC = 0.9317
	- test confusion matrix = TN 97, FP 18, FN 10, TP 86
- Follow-up tuned-run training-curve interpretation:
	- increasing the learning rate from 1e-4 to 5e-4 while keeping dropout and weight decay at 0.3 / 5e-4 materially improved all three saved checkpoints relative to experiment_002
	- small_mlp no longer behaved like a model that was still underfitting all the way to epoch 100; its best validation-loss checkpoint moved from epoch 100 in experiment_002 to epoch 47 here, and both test F1 and ROC-AUC recovered substantially
	- recommended_mlp also recovered from the overly conservative regime in experiment_002, but its best validation-loss checkpoint still occurred early at epoch 19 and the remaining epochs mostly added overfitting rather than useful learning
	- large_mlp remained the strongest architecture under the 5e-4 learning-rate regime, but its best checkpoint occurred even earlier at epoch 17, which shows the 150-epoch budget was much larger than needed for the selected checkpoint
	- across all three models, extending the budget from 100 to 150 epochs mainly created a long late-epoch overfitting tail; because the notebook restores the best validation-loss checkpoint, the exported results improved even though the final-epoch curves look markedly worse
- Direct comparison versus experiment_002:
	- all three architectures improved on the held-out test set after increasing the learning rate to 5e-4
	- small_mlp improved materially: test F1 rose from 0.8211 to 0.8426, test ROC-AUC rose from 0.9241 to 0.9358, and false negatives dropped from 18 to 13, although false positives rose from 16 to 18
	- recommended_mlp also improved: test F1 rose from 0.8168 to 0.8351 and false negatives dropped from 18 to 15 while false positives stayed at 17
	- large_mlp improved modestly but remained strongest: test accuracy rose from 0.8626 to 0.8673, test F1 rose from 0.8528 to 0.8600, and recall rose from 0.8750 to 0.8958 while false positives rose slightly from 17 to 18
	- these results support the conclusion that 1e-4 was too conservative and that 5e-4 is the stronger learning-rate regime under the current AdamW + dropout 0.3 + weight decay 5e-4 setup
- Direct comparison versus experiment_001:
	- small_mlp and recommended_mlp recovered part of the performance lost in experiment_002, but both still remained slightly below their baseline experiment_001 test F1 results
	- large_mlp now outperformed its baseline clearly: test F1 rose from 0.8342 in experiment_001 to 0.8600 here and test accuracy rose from 0.8436 to 0.8673
	- the architecture ranking under the stronger regularization regime therefore remained different from the baseline run: large_mlp stayed strongest, while small_mlp no longer held the best thresholded test performance
- Follow-up tuned-run comparison takeaway:
	- experiment_003 answered the experiment_002 follow-up question positively: a middle learning rate of 5e-4 retained the large-model gains and materially recovered the smaller models relative to experiment_002
	- however, the extra 50 epochs were not useful for selected-checkpoint quality; every best checkpoint occurred by epoch 47, so future runs under this regime do not need a 150-epoch budget
	- the strongest current recorded model is large_mlp under the 5e-4 learning-rate regime, with test F1 = 0.8600 and test accuracy = 0.8673
	- this run also exposes a policy mismatch for future tuning: the notebook currently restores the best validation-loss checkpoint, while the thesis development rule states that hyperparameter choices should be guided by validation F1

# Midpoint-Dropout Follow-up Run: experiment_004

- Follow-up tuned run executed and exported with these run metadata:
	- run directory = results/experiment_004
	- run timestamp = 2026-05-02T17:46:03
	- device = cpu
	- torch version = 2.11.0+cpu
	- CUDA available = false
- Global notebook defaults used for this run:
	- dropout rate = 0.25
	- learning rate = 5e-4
	- weight decay = 5e-4
	- max epochs = 100
- Exported follow-up artifacts created:
	- results/experiment_004/summary.csv
	- results/experiment_004/summary.json
	- results/experiment_004/test_confusion_summary.csv
	- results/experiment_004/history/*.csv
	- results/experiment_004/plots/*.png
- Follow-up tuned run results, architecture = 71 -> 32 -> 1 for small_mlp:
	- epochs trained = 100
	- best epoch = 47
	- validation loss = 0.3850
	- validation accuracy = 0.8389
	- validation F1 = 0.8211
	- validation ROC-AUC = 0.9111
	- test loss = 0.3197
	- test accuracy = 0.8578
	- test precision = 0.8300
	- test recall = 0.8646
	- test F1 = 0.8469
	- test ROC-AUC = 0.9385
	- test confusion matrix = TN 98, FP 17, FN 13, TP 83
- Follow-up tuned run results, architecture = 71 -> 64 -> 32 -> 1 for recommended_mlp:
	- epochs trained = 100
	- best epoch = 28
	- validation loss = 0.3859
	- validation accuracy = 0.8578
	- validation F1 = 0.8454
	- validation ROC-AUC = 0.9158
	- test loss = 0.3207
	- test accuracy = 0.8673
	- test precision = 0.8400
	- test recall = 0.8750
	- test F1 = 0.8571
	- test ROC-AUC = 0.9346
	- test confusion matrix = TN 99, FP 16, FN 12, TP 84
- Follow-up tuned run results, architecture = 71 -> 128 -> 64 -> 32 -> 1 for large_mlp:
	- epochs trained = 100
	- best epoch = 18
	- validation loss = 0.3743
	- validation accuracy = 0.8673
	- validation F1 = 0.8571
	- validation ROC-AUC = 0.9171
	- test loss = 0.3448
	- test accuracy = 0.8483
	- test precision = 0.8137
	- test recall = 0.8646
	- test F1 = 0.8384
	- test ROC-AUC = 0.9244
	- test confusion matrix = TN 96, FP 19, FN 13, TP 83
- Follow-up tuned-run training-curve interpretation:
	- lowering dropout from 0.30 to 0.25 while keeping the 5e-4 learning rate and 5e-4 weight decay changed the architecture ranking again
	- small_mlp improved slightly over experiment_003 and still showed only mild late-epoch overfitting: its best checkpoint stayed at epoch 47, while validation loss rose from 0.3850 at the selected checkpoint to 0.4071 at epoch 100 and validation F1 slipped from 0.8211 to 0.8000
	- recommended_mlp benefited the most from the lighter dropout setting: its best validation-loss checkpoint moved from epoch 19 in experiment_003 to epoch 28 here, and it became the strongest model within experiment_004 on held-out test accuracy, precision, F1, and total error count
	- large_mlp lost ground under the lighter dropout setting: its best checkpoint still occurred early at epoch 18, but validation loss then deteriorated sharply to 0.6101 by epoch 100 while validation F1 fell from 0.8571 to 0.7667, which indicates the strongest overfitting of the three models
	- overall, dropout 0.25 appears to be a better regularization point for recommended_mlp, a small improvement for small_mlp, and too weak for large_mlp under the current AdamW + 5e-4 learning-rate regime
- Direct comparison versus experiment_003:
	- small_mlp improved slightly: test F1 rose from 0.8426 to 0.8469, test accuracy rose from 0.8531 to 0.8578, test ROC-AUC rose from 0.9358 to 0.9385, and test loss improved from 0.3228 to 0.3197
	- recommended_mlp improved materially: test F1 rose from 0.8351 to 0.8571, test accuracy rose from 0.8483 to 0.8673, test loss improved from 0.3336 to 0.3207, and test ROC-AUC edged up from 0.9330 to 0.9346
	- large_mlp regressed materially: test F1 fell from 0.8600 to 0.8384, test accuracy fell from 0.8673 to 0.8483, test ROC-AUC fell from 0.9317 to 0.9244, and test loss worsened from 0.3256 to 0.3448
	- these results show that lowering dropout from 0.30 to 0.25 redistributed useful capacity toward the small and recommended models but removed too much regularization from the large model
- Direct comparison versus experiment_001:
	- small_mlp remained below its baseline experiment_001 thresholded test performance even after the recovery from experiment_002: test F1 here was 0.8469 versus 0.8629 in experiment_001
	- recommended_mlp now exceeded its baseline clearly: test F1 rose from 0.8410 in experiment_001 to 0.8571 here and test accuracy rose from 0.8531 to 0.8673, while test ROC-AUC stayed effectively unchanged at 0.9346 versus 0.9349
	- large_mlp remained slightly above its baseline experiment_001 test F1, 0.8384 versus 0.8342, but gave back most of the large-model gains that had appeared in experiment_003
	- the best overall model across experiment_001 through experiment_004 still did not change unambiguously: experiment_001 small_mlp retained the highest recorded test F1 at 0.8629, while experiment_003 large_mlp remained very close at 0.8600 and experiment_004 recommended_mlp entered the top tier at 0.8571
- Midpoint-dropout follow-up takeaway:
	- experiment_004 confirms again that architecture ranking in this project is highly sensitive to regularization strength
	- within this run, recommended_mlp is the strongest model by test accuracy, precision, F1, and balanced total errors, while small_mlp retains the best test loss and ROC-AUC
	- the 0.25 dropout setting is a promising direction for recommended_mlp specifically, but it is not a universal improvement across architectures because large_mlp degraded substantially
	- for comparability with experiment_001 through experiment_003, experiment_004 intentionally kept the same validation-loss checkpoint selection rule rather than switching the notebook mid-sweep to validation-F1 checkpoint restoration

# Model-Selection Metric Decision

- Primary model-selection metric for the thesis: test F1
- Main secondary metric: test ROC-AUC
- Additional reported metrics: test precision, test recall, and test accuracy
- Support metric: test loss, used mainly to discuss probability quality, calibration drift, and checkpoint behavior rather than to choose the best classifier
- Development rule for future tuning runs: choose hyperparameters by validation F1 and keep the held-out test set for the final locked comparison only
- Comparability note for the first official sweep, experiment_001 through experiment_004: checkpoint restoration remained based on minimum validation loss so those four runs stay directly comparable under the same notebook selection rule
- Decision rationale:
	- the thesis task is framed as binary active versus inactive classification with a fixed decision threshold, so the primary metric should reward good thresholded classification behavior rather than ranking alone
	- F1 is the most appropriate single summary because it balances precision and recall for the active class, which is the scientifically relevant positive class in this screening context
	- accuracy is still useful to report, but it is too coarse to be the main selection rule because it can hide the tradeoff between false positives and false negatives
	- ROC-AUC remains important because it measures ranking quality independently of the classification threshold, but a model can obtain a slightly better ROC-AUC while still making worse final binary decisions at the chosen threshold
	- test loss is informative about confidence and calibration, especially when validation loss rises while F1 remains relatively stable, but it is less directly aligned with the practical question of which model classifies compounds better at the final threshold
	- this choice remains compatible with the observed run-to-run behavior: in experiment_001 the strongest thresholded classifier was small_mlp, while in the later tuned experiment_002 and experiment_003 runs the strongest thresholded classifier became large_mlp
- Thesis-ready wording for the report:
	- The primary model-selection metric in this study is the test-set F1 score. This choice is justified because the task is formulated as a binary classification problem in which compounds are assigned to the active or inactive class using a fixed decision threshold. In this setting, the most relevant practical objective is not only to maximize the total number of correct predictions, but to maintain a balanced tradeoff between precision and recall for the active class, which is the class of greatest interest for compound prioritization. For that reason, F1 provides a more informative single summary than accuracy, which can obscure the balance between false positives and false negatives, and a more decision-aligned criterion than ROC-AUC, which evaluates ranking quality independently of the final threshold. ROC-AUC is therefore retained as a secondary metric to assess ranking performance, while test loss is used as a supportive indicator of calibration and confidence rather than as the main criterion for choosing the best model.
- Caveat for future iterations:
	- if the thesis later adopts a domain rule that missing an active compound is substantially worse than advancing a false positive, recall or a recall-constrained metric should be reconsidered as the primary decision criterion

# Planned Classical ML Baseline Sweep

- Purpose: establish the first official classical machine-learning baseline that is directly comparable to the current MLP experiments.
- Frozen comparison setup:
	- use the exact same final tabular input as the deep learning notebook: datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/train_dataset.csv, validation_dataset.csv, and test_dataset.csv
	- do not add new preprocessing, new feature selection, SMOTE, or train-plus-validation retraining in this first baseline study so the comparison remains fair against the current deep learning runs
- Model families included in the sweep:
	- Logistic Regression
	- Random Forest
	- SVM
- Planned hyperparameter scope:
	- Logistic Regression: regularization strength C and optional class_weight
	- Random Forest: number of trees, max_depth, minimum samples per leaf, and optional class_weight
	- SVM: kernel, C, gamma for the nonlinear case, and optional class_weight
	- the grid will stay modest on purpose so the baseline is strong and defensible without turning baseline selection into a separate thesis-scale optimization study
- Training and selection protocol:
	- fit every candidate configuration on the training split only
	- evaluate every candidate configuration on the validation split only
	- use validation F1 as the primary selection metric
	- use validation ROC-AUC as the main secondary selection metric
	- keep the default decision threshold at 0.5 for comparability with the current deep learning experiments
- Freezing rule before the held-out test comparison:
	- choose one validation-best configuration inside each model family by validation F1
	- compare the three family winners by validation F1
	- freeze one official classical baseline model before touching the test set
- Final held-out evaluation for the frozen baseline:
	- evaluate the selected baseline once on the test split
	- report test F1 as the primary metric
	- report test ROC-AUC as the main secondary metric
	- also report test precision, test recall, test accuracy, confusion matrix, and log loss as supporting outputs
- Planned reproducible artifacts:
	- a full table of all validation trials
	- a summary of the selected classical baseline
	- the final held-out test metrics
	- the final test confusion matrix
- Thesis note:
	- this first classical baseline study prioritizes direct comparability with the current MLP workflow over exhaustive classical-model tuning

# Classical ML Baseline Run: experiment_001

- Notebook created: classical_ml_baseline_pipeline.ipynb
- Run executed and exported with these run metadata:
	- run directory = results/classical_ml_baseline/experiment_001
	- run timestamp recorded in results/classical_ml_baseline/experiment_001/summary.json
	- dataset source = datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/
	- decision threshold = 0.5
	- selection protocol = train on the training split only, select by validation F1, use validation ROC-AUC as the main secondary metric, and evaluate the frozen winner once on test
- Exported baseline artifacts created:
	- results/classical_ml_baseline/experiment_001/validation_trials.csv
	- results/classical_ml_baseline/experiment_001/family_winners.csv
	- results/classical_ml_baseline/experiment_001/selected_baseline_summary.csv
	- results/classical_ml_baseline/experiment_001/test_confusion_summary.csv
	- results/classical_ml_baseline/experiment_001/summary.json
	- results/classical_ml_baseline/experiment_001/plots/selected_baseline_test_confusion_matrix.png
- Validation-best family winners:
	- Random Forest winner: class_weight = balanced, n_estimators = 200, max_depth = 10, min_samples_leaf = 1, validation F1 = 0.8718, validation ROC-AUC = 0.9471, validation log loss = 0.3178
	- SVM winner: kernel = rbf, C = 10.0, gamma = scale, class_weight = None, validation F1 = 0.8543, validation ROC-AUC = 0.9274, validation log loss = 0.3458
	- Logistic Regression winner: C = 1.0, class_weight = balanced, validation F1 = 0.7817, validation ROC-AUC = 0.8643, validation log loss = 0.5172
- Frozen official classical baseline selected from validation only:
	- model family = Random Forest
	- hyperparameters = class_weight balanced, n_estimators 200, max_depth 10, min_samples_leaf 1
- Frozen baseline test-set results:
	- test accuracy = 0.8531
	- test precision = 0.7928
	- test recall = 0.9167
	- test F1 = 0.8502
	- test ROC-AUC = 0.9398
	- test log loss = 0.3376
	- test confusion matrix = TN 92, FP 23, FN 8, TP 88
- First comparison takeaway versus the current deep-learning runs:
	- the classical Random Forest baseline is competitive and clearly stronger than Logistic Regression and SVM on this descriptor-based split
	- on the thesis primary metric, test F1, the strongest recorded deep-learning runs still remain ahead: experiment_001 small_mlp = 0.8629, experiment_003 large_mlp = 0.8600, and experiment_004 recommended_mlp = 0.8571, versus 0.8502 for the Random Forest baseline
	- the Random Forest baseline currently has the strongest recorded ranking metric, test ROC-AUC = 0.9398, which is slightly above the best recorded deep-learning ROC-AUC values so far
	- the Random Forest baseline also achieved very high recall, 0.9167, which is useful to discuss if the thesis later emphasizes sensitivity to active compounds
	- the current evidence therefore supports the thesis comparison framing: deep learning is not trivially dominant, but the strongest MLP checkpoints still hold a small lead on the primary thresholded-classification metric

# Deep Learning vs Classical ML TL;DR

- Best classical ML model so far: Random Forest baseline.
- Best Random Forest test results: F1 = 0.8502, ROC-AUC = 0.9398, recall = 0.9167.
- Best deep-learning test F1 results so far:
	- experiment_001 small_mlp = 0.8629
	- experiment_003 large_mlp = 0.8600
	- experiment_004 recommended_mlp = 0.8571
- Simple comparison summary:
	- on the thesis primary metric, test F1, the best MLP models are slightly better than the Random Forest baseline
	- on the main ranking metric, test ROC-AUC, the Random Forest baseline is slightly better than the best current MLP runs
	- the correct overall conclusion is not that deep learning clearly crushes classical ML; instead, deep learning currently offers a small edge in final thresholded classification, while classical ML remains very competitive
	- if the thesis later emphasizes sensitivity to active compounds, the Random Forest baseline deserves discussion because it currently has the highest recall among the compared locked runs

# Open Preprocessing Decisions

- Split the dataset before fitting variance or correlation filters so those preprocessing choices are learned from the training rows only and do not leak information from validation or test rows
- Zero-variance filtering should use threshold 0.0 at this stage, which means remove only descriptor columns that are completely constant and therefore contain no information
- The first modeling split will use a stratified 80/10/10 train/validation/test partition so the active/inactive balance remains stable across all three subsets
- Apply normalization or scaling only where required by the chosen model family and fit it on the training split only
- Keep 300 nM as a possible sensitivity-analysis threshold if comparison with a fixed domain-informed cutoff is needed later

# EC50 Dataset Snapshot

- Date: 2026-05-02
- Source file: datasets/CHEMBL379_EC50_AllDesc.csv
- Total rows: 8664
- All rows correspond to EC50 measurements in nM for target CHEMBL379
- Unique Molecule ChEMBL IDs: 8664
- Unique SMILES strings: 8664
- Standard Relation distribution:
	- '=' = 7336
	- '>' = 1048
	- '<' = 276
	- '>=' = 3
	- '<=' = 1
- Unlike the IC50 export, the EC50 export includes four inclusive-bound rows ('>=' and '<='), so the mirrored curation logic must handle those relations explicitly instead of silently excluding them

# EC50 Labeling Policy v1

- The binary activity label is derived from EC50 values using a threshold computed from exact measurements only
- Censored values with relations '>', '>=', '<', and '<=' are excluded from threshold estimation because they are bounds rather than exact values
- Median EC50 using only exact rows ('=') is 490 nM
- Definition:
	- activity = 1 if EC50 <= 490 nM
	- activity = 0 if EC50 > 490 nM
- Row-level labeling rules:
	- If Standard Relation is '=': assign the label directly from Standard Value
	- If Standard Relation is '>' or '>=':
		- assign activity = 0 when the reported bound is >= 490 nM
		- exclude the row if the reported bound is < 490 nM because the true EC50 remains ambiguous with respect to the threshold
	- If Standard Relation is '<' or '<=':
		- assign activity = 1 when the reported bound is <= 490 nM
		- exclude the row if the reported bound is > 490 nM because the true EC50 remains ambiguous with respect to the threshold
- Counts under this policy:
	- Exact rows labeled active: 3682
	- Exact rows labeled inactive: 3654
	- '>' or '>=' rows safely labeled inactive: 1015
	- '<' or '<=' rows safely labeled active: 256
	- Ambiguous censored rows excluded: 57
	- Total rows retained after labeling: 8607
	- Final active rows: 3938
	- Final inactive rows: 4669

# EC50 Classification Dataset Builder

- Script created: datasets/postprocessed-CHEMBL379_EC50/build_classification_dataset.py
- Shared helper module created: datasets/postprocessed-CHEMBL379_EC50/common.py
- Purpose: build the clean base dataset for binary EC50 classification before descriptor regeneration, split creation, and later feature filtering
- Default output folder: datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset/
- The builder accepts optional `--output-dir` and `--suffix` arguments so comparison runs can write to a separate folder without overwriting prior artifacts
- The script performs these steps:
	- read the raw tab-delimited CHEMBL379 EC50 export
	- keep only rows within thesis scope: EC50, nM, CHEMBL379
	- compute the 490 nM threshold from exact rows only
	- assign the binary activity label using the censor-aware policy, including the EC50-specific inclusive bounds '>=' and '<='
	- exclude ambiguous censored rows
	- group rows by case-sensitive SMILES to detect exact duplicates
	- keep one representative row only if an exact duplicate group is label-concordant
	- exclude a group only if an exact duplicate group is label-conflicting
- In the current dataset snapshot, the duplicate-handling branch did not alter the data because no exact duplicate SMILES were found
- Main dataset created: datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset/classification_base_dataset.csv
- Audit file created: datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset/excluded_ambiguous_censored.csv
- Audit file created: datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset/collapsed_concordant_duplicates.csv
- Audit file created: datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset/excluded_conflicting_duplicates.csv
- Summary file created: datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset/classification_curation_summary.json
- Current generated counts:
	- input rows: 8664
	- threshold: 490.0 nM
	- labeled rows after censor-aware filtering: 8607
	- ambiguous censored rows excluded: 57
	- collapsed duplicate groups: 0
	- conflicting duplicate groups: 0
	- final classification rows: 8607
	- final active rows: 3938
	- final inactive rows: 4669
	- Safe validation rerun also completed successfully into datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset_validation/ with the same counts, confirming that `--suffix validation` avoids overwriting the default artifacts

# EC50 RDKit Descriptor Regeneration

- Script created: datasets/postprocessed-CHEMBL379_EC50/build_rdkit_descriptor_dataset.py
- Shared preprocessing helper module created: datasets/postprocessed-CHEMBL379_EC50/preprocessing_common.py
- Purpose: regenerate the descriptor matrix directly from SMILES for the clean EC50 classification dataset instead of relying on the original exported descriptor values
- Default classification input: datasets/postprocessed-CHEMBL379_EC50/classification_curation_dataset/classification_base_dataset.csv
- Main dataset created: datasets/postprocessed-CHEMBL379_EC50/rdkit_descriptor_dataset/classification_rdkit_descriptor_dataset.csv
- Audit file created: datasets/postprocessed-CHEMBL379_EC50/rdkit_descriptor_dataset/rdkit_invalid_smiles.csv
- Audit file created: datasets/postprocessed-CHEMBL379_EC50/rdkit_descriptor_dataset/rdkit_canonical_smiles_collisions.csv
- Summary file created: datasets/postprocessed-CHEMBL379_EC50/rdkit_descriptor_dataset/rdkit_descriptor_summary.json
- Current regenerated descriptor counts:
	- input classification rows: 8607
	- descriptor rows written: 8607
	- invalid SMILES rows: 0
	- canonical collision groups: 0
	- descriptor column count: 119

# EC50 Modeling Split

- Script created: datasets/postprocessed-CHEMBL379_EC50/build_stratified_split_dataset.py
- Purpose: create the fixed train/validation/test split for the EC50 track before later filtering and scaling stages
- Default input dataset: datasets/postprocessed-CHEMBL379_EC50/rdkit_descriptor_dataset/classification_rdkit_descriptor_dataset.csv
- Default output folder: datasets/postprocessed-CHEMBL379_EC50/stratified_split_dataset/
- Current split policy:
	- stratified by binary activity label
	- train fraction = 0.8
	- validation fraction = 0.1
	- test fraction = 0.1
	- random seed = 42
- Current split counts:
	- total rows assigned: 8607
	- train rows: 6885
	- validation rows: 861
	- test rows: 861
	- train active/inactive: 3150 / 3735
	- validation active/inactive: 394 / 467
	- test active/inactive: 394 / 467

# EC50 Variance Filtering

- Script created: datasets/postprocessed-CHEMBL379_EC50/build_variance_filtered_dataset.py
- Purpose: fit descriptor variance filtering on the EC50 training split only and apply the same kept-column set to validation and test without leakage
- Default input folder: datasets/postprocessed-CHEMBL379_EC50/stratified_split_dataset/
- Default output folder: datasets/postprocessed-CHEMBL379_EC50/variance_filtered_dataset/
- Current variance filter result:
	- input descriptor count: 119
	- kept descriptor count: 114
	- dropped descriptor count: 5
	- dropped descriptors:
		- slogp_VSA9
		- smr_VSA8
		- MQN18
		- MQN22
		- MQN23
	- train rows after filtering: 6885
	- validation rows after filtering: 861
	- test rows after filtering: 861

# EC50 Correlation Filtering

- Script created: datasets/postprocessed-CHEMBL379_EC50/build_correlation_filtered_dataset.py
- Purpose: fit descriptor correlation filtering on the EC50 training split only and apply the same kept-column set to validation and test without leakage
- Default input folder: datasets/postprocessed-CHEMBL379_EC50/variance_filtered_dataset/
- Default output folder: datasets/postprocessed-CHEMBL379_EC50/correlation_filtered_dataset/
- Current correlation filter result:
	- input descriptor count: 114
	- kept descriptor count: 74
	- dropped descriptor count: 40
	- train rows after filtering: 6885
	- validation rows after filtering: 861
	- test rows after filtering: 861

# EC50 Scaling

- Script created: datasets/postprocessed-CHEMBL379_EC50/build_scaled_dataset.py
- Purpose: fit descriptor scaling on the EC50 training split only after correlation filtering and apply the same descriptor-wise statistics to validation and test
- Default input folder: datasets/postprocessed-CHEMBL379_EC50/correlation_filtered_dataset/
- Default output folder: datasets/postprocessed-CHEMBL379_EC50/scaled_dataset/
- Current scaling result:
	- scaled descriptor count: 74
	- train rows after scaling: 6885
	- validation rows after scaling: 861
	- test rows after scaling: 861

# Large-MLP Notebook Track Selector

- Notebook updated: large_mlp_optimization_pipeline.ipynb
- The first setup cell now exposes a single `DATASET_TRACK` variable with supported values `IC50` and `EC50`
- The notebook derives both `DATA_DIR` and `RESULTS_ROOT` from that track selection so input data and exported artifacts stay aligned
- Track-specific results roots are now:
	- results/large_mlp_optimization/IC50/
	- results/large_mlp_optimization/EC50/
- Validation completed for the default IC50 setting by rerunning the notebook setup, run-directory, and load-data cells successfully
- Current validated IC50 load after the refactor:
	- train shape = (1686, 82)
	- validation shape = (211, 82)
	- test shape = (211, 82)
- Operational rule for future runs:
	- change only `DATASET_TRACK`
	- rerun Cells 1 through 8 before training or exporting so the tensors, loaders, input dimension, and output directory match the selected track

# Deep-Learning And Classical-ML Track Selector

- Notebooks updated:
	- deep_learning_pipeline.ipynb
	- classical_ml_baseline_pipeline.ipynb
- Both notebooks now expose a single `DATASET_TRACK` variable with supported values `IC50` and `EC50`
- Both notebooks now derive `DATA_DIR` and `RESULTS_ROOT` from that track selection so dataset inputs and exported artifacts stay aligned
- Track-specific results roots are now:
	- results/deep_learning_pipeline/IC50/
	- results/deep_learning_pipeline/EC50/
	- results/classical_ml_baseline/IC50/
	- results/classical_ml_baseline/EC50/
- Validation completed for the default IC50 setting by rerunning the setup, run-directory, and load-data cells in both notebooks successfully
- Current validated IC50 routing after the refactor:
	- deep_learning_pipeline.ipynb loads datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/ and writes under results/deep_learning_pipeline/IC50/
	- classical_ml_baseline_pipeline.ipynb loads datasets/postprocessed-CHEMBL379_IC50/scaled_dataset/ and writes under results/classical_ml_baseline/IC50/
- Operational rule for future runs:
	- change only `DATASET_TRACK`
	- rerun the first setup/load cells before training or exporting so the active dataset and results folder stay synchronized

# EC50 Deep-Learning Baseline Run: experiment_001

- Notebook used: deep_learning_pipeline.ipynb
- Run executed and exported with these run metadata:
	- run directory = results/deep_learning_pipeline/EC50/experiment_001
	- run timestamp = 2026-05-02T22:18:54
	- dataset source = datasets/postprocessed-CHEMBL379_EC50/scaled_dataset/
	- device = cpu
	- torch version = 2.11.0+cpu
	- CUDA available = false
	- checkpoint policy = train the full 100 epochs, then restore the checkpoint with minimum validation loss
- Global notebook defaults used for this run:
	- dropout rate = 0.2
	- learning rate = 1e-3
	- weight decay = 1e-4
	- max epochs = 100
- Exported EC50 deep-learning artifacts created:
	- results/deep_learning_pipeline/EC50/experiment_001/summary.csv
	- results/deep_learning_pipeline/EC50/experiment_001/summary.json
	- results/deep_learning_pipeline/EC50/experiment_001/test_confusion_summary.csv
	- results/deep_learning_pipeline/EC50/experiment_001/history/*.csv
	- results/deep_learning_pipeline/EC50/experiment_001/plots/*.png
- EC50 deep-learning results, architecture = 74 -> 32 -> 1 for small_mlp:
	- epochs trained = 100
	- best epoch = 82
	- validation loss = 0.3496
	- validation accuracy = 0.8537
	- validation precision = 0.8661
	- validation recall = 0.8046
	- validation F1 = 0.8342
	- validation ROC-AUC = 0.9249
	- test loss = 0.3443
	- test accuracy = 0.8548
	- test precision = 0.8726
	- test recall = 0.7995
	- test F1 = 0.8344
	- test ROC-AUC = 0.9288
	- test confusion matrix = TN 421, FP 46, FN 79, TP 315
- EC50 deep-learning results, architecture = 74 -> 64 -> 32 -> 1 for recommended_mlp:
	- epochs trained = 100
	- best epoch = 75
	- validation loss = 0.3041
	- validation accuracy = 0.8653
	- validation precision = 0.8546
	- validation recall = 0.8503
	- validation F1 = 0.8524
	- validation ROC-AUC = 0.9435
	- test loss = 0.3686
	- test accuracy = 0.8746
	- test precision = 0.8629
	- test recall = 0.8629
	- test F1 = 0.8629
	- test ROC-AUC = 0.9280
	- test confusion matrix = TN 413, FP 54, FN 54, TP 340
- EC50 deep-learning results, architecture = 74 -> 128 -> 64 -> 32 -> 1 for large_mlp:
	- epochs trained = 100
	- best epoch = 44
	- validation loss = 0.3331
	- validation accuracy = 0.8606
	- validation precision = 0.8309
	- validation recall = 0.8731
	- validation F1 = 0.8515
	- validation ROC-AUC = 0.9374
	- test loss = 0.3406
	- test accuracy = 0.8653
	- test precision = 0.8458
	- test recall = 0.8629
	- test F1 = 0.8543
	- test ROC-AUC = 0.9339
	- test confusion matrix = TN 405, FP 62, FN 54, TP 340
- EC50 deep-learning training-curve interpretation:
	- all three models completed the full 100 epochs, so the no-early-stop plus best-checkpoint policy behaved as intended on the EC50 track as well
	- small_mlp showed the mildest late overfitting: validation loss bottomed at epoch 82 and increased by only about 0.0076 by epoch 100, while validation F1 and ROC-AUC both peaked later at epoch 90 and stayed on a broad plateau; this is the most conservative classifier of the three, with the highest precision and specificity but also the most false negatives
	- recommended_mlp produced the strongest overall validation checkpoint and the strongest held-out thresholded test performance; its validation-loss minimum occurred at epoch 75, but the best validation F1 appeared earlier at epoch 58 and the best validation ROC-AUC later at epoch 90, which indicates a wide stable operating region rather than a sharp optimum
	- large_mlp began to overfit earliest in loss terms: validation loss bottomed at epoch 44 and rose by about 0.0364 by epoch 100, but validation F1 and ROC-AUC remained comparatively stable, which suggests calibration drift more than a collapse in class separation
	- across all three models, late-epoch degradation is milder on EC50 than in the original IC50 baseline run: the gap between the selected checkpoint and epoch 100 is modest in validation F1 for every architecture, so checkpoint restoration remains important mainly for calibration and loss quality rather than for rescuing a failing classifier
- EC50 deep-learning comparison takeaway within experiment_001:
	- recommended_mlp is the strongest model on the thesis primary metric, test F1, and also on test accuracy and total error count; it made the fewest total mistakes, 108 out of 861 test samples, with a nearly perfectly balanced precision and recall profile
	- small_mlp is the most conservative classifier: it achieved the highest test precision, 0.8726, and the highest specificity, but that came with the lowest recall and by far the most false negatives, 79, so it is the weakest option if missing active compounds matters
	- large_mlp achieved the best test ROC-AUC, 0.9339, and the best test loss, 0.3406, but it paid for that ranking-oriented strength with more false positives than recommended_mlp, 62 versus 54, so it lost on thresholded accuracy and F1 at the fixed 0.5 decision threshold
	- this run again matches the thesis metric-policy rationale: the model with the best ranking or probability-quality metric is not automatically the model with the best final classification decisions at the chosen threshold
- Direct comparison versus IC50 deep-learning experiment_001:
	- small_mlp regressed noticeably on EC50 relative to IC50: test F1 fell from 0.8629 to 0.8344, test accuracy fell from 0.8720 to 0.8548, test ROC-AUC fell from 0.9341 to 0.9288, and test loss worsened from 0.3279 to 0.3443
	- recommended_mlp improved materially on the thesis primary metric: test F1 rose from 0.8410 on IC50 to 0.8629 on EC50 and test accuracy rose from 0.8531 to 0.8746, although test ROC-AUC decreased from 0.9349 to 0.9280 and test loss worsened from 0.3200 to 0.3686
	- large_mlp also improved materially on EC50 relative to IC50: test F1 rose from 0.8342 to 0.8543, test accuracy rose from 0.8436 to 0.8653, and test ROC-AUC rose from 0.9297 to 0.9339, while test loss worsened only slightly from 0.3368 to 0.3406
	- the architecture ranking therefore changes between the IC50 and EC50 tracks even under the same baseline notebook defaults: small_mlp was the strongest thresholded model in IC50 experiment_001, whereas recommended_mlp is the strongest thresholded model in EC50 experiment_001
- Cross-check against the EC50 optimized-large probe:
	- the standard deep-learning large_mlp is effectively tied with the specialized optimized-large architecture on EC50 for thresholded performance, 0.8543 versus 0.8541 test F1, and slightly edges it on test ROC-AUC, 0.9339 versus 0.9333
	- however, recommended_mlp from the standard deep-learning notebook now sets the strongest EC50 thresholded result recorded so far, with test F1 = 0.8629 and test accuracy = 0.8746, even though the optimized-large model still retains slightly better probability-quality metrics than recommended_mlp
- EC50 deep-learning baseline takeaway:
	- the first official EC50 deep-learning baseline is strong and scientifically useful: it confirms that EC50 is not merely a duplicate track of IC50 because the best baseline architecture changes and the error tradeoffs shift materially
	- under the thesis primary metric, recommended_mlp is the EC50 baseline winner and should be treated as the default EC50 deep-learning reference point unless a later tuned EC50 run surpasses it
	- the next fair comparison step is now clear: run the classical ML baseline notebook on the EC50 track and compare the locked IC50 and EC50 classical versus deep-learning baselines under the same metric policy

# EC50 Tuned Deep-Learning Run: experiment_002

- Notebook used: deep_learning_pipeline.ipynb
- Tuned run executed and exported with these run metadata:
	- run directory = results/deep_learning_pipeline/EC50/experiment_002
	- run timestamp = 2026-05-02T23:47:22
	- dataset source = datasets/postprocessed-CHEMBL379_EC50/scaled_dataset/
	- device = cpu
	- torch version = 2.11.0+cpu
	- CUDA available = false
	- checkpoint policy = train the full 100 epochs, then restore the checkpoint with minimum validation loss
- Global notebook defaults used for this run:
	- dropout rate = 0.3
	- learning rate = 1e-4
	- weight decay = 5e-4
	- max epochs = 100
- Exported tuned EC50 artifacts created:
	- results/deep_learning_pipeline/EC50/experiment_002/summary.csv
	- results/deep_learning_pipeline/EC50/experiment_002/summary.json
	- results/deep_learning_pipeline/EC50/experiment_002/test_confusion_summary.csv
	- results/deep_learning_pipeline/EC50/experiment_002/history/*.csv
	- results/deep_learning_pipeline/EC50/experiment_002/plots/*.png
- Tuned EC50 results, architecture = 74 -> 32 -> 1 for small_mlp:
	- epochs trained = 100
	- best epoch = 100
	- validation loss = 0.3960
	- validation accuracy = 0.8258
	- validation precision = 0.8315
	- validation recall = 0.7766
	- validation F1 = 0.8031
	- validation ROC-AUC = 0.9038
	- test loss = 0.4082
	- test accuracy = 0.8084
	- test precision = 0.8172
	- test recall = 0.7487
	- test F1 = 0.7815
	- test ROC-AUC = 0.8949
	- test confusion matrix = TN 401, FP 66, FN 99, TP 295
- Tuned EC50 results, architecture = 74 -> 64 -> 32 -> 1 for recommended_mlp:
	- epochs trained = 100
	- best epoch = 99
	- validation loss = 0.3651
	- validation accuracy = 0.8444
	- validation precision = 0.8385
	- validation recall = 0.8173
	- validation F1 = 0.8278
	- validation ROC-AUC = 0.9168
	- test loss = 0.3584
	- test accuracy = 0.8374
	- test precision = 0.8378
	- test recall = 0.7995
	- test F1 = 0.8182
	- test ROC-AUC = 0.9201
	- test confusion matrix = TN 406, FP 61, FN 79, TP 315
- Tuned EC50 results, architecture = 74 -> 128 -> 64 -> 32 -> 1 for large_mlp:
	- epochs trained = 100
	- best epoch = 98
	- validation loss = 0.3438
	- validation accuracy = 0.8525
	- validation precision = 0.8346
	- validation recall = 0.8452
	- validation F1 = 0.8398
	- validation ROC-AUC = 0.9266
	- test loss = 0.3266
	- test accuracy = 0.8676
	- test precision = 0.8571
	- test recall = 0.8528
	- test F1 = 0.8550
	- test ROC-AUC = 0.9340
	- test confusion matrix = TN 411, FP 56, FN 58, TP 336
- Tuned EC50 training-curve interpretation:
	- this run reproduces the same high-regularization, low-learning-rate regime that had been tested earlier on IC50, and the EC50 curves show the same core pattern: the optimization became much more conservative and the best checkpoints moved to the very end of training
	- small_mlp no longer shows the mild late overfitting seen in EC50 experiment_001; instead, validation loss keeps decreasing almost monotonically through epoch 100, the best validation-loss checkpoint is the final epoch, and the best validation ROC-AUC is also the final epoch. This indicates the 1e-4 regime is too conservative for this model under a 100-epoch budget and likely leaves it undertrained rather than overfit
	- recommended_mlp behaves similarly: validation loss continues improving through epoch 99, late-epoch degradation is almost absent, and the best validation-loss and ROC-AUC checkpoints are both effectively at the end. This again points to under-optimization rather than excess capacity
	- large_mlp benefits the most from this regime on EC50, just as it did on IC50: its best validation-loss checkpoint is epoch 98 and its best validation F1 and ROC-AUC occur at epoch 100, so even the largest model still had not clearly exhausted learning by the end of the run
	- because all three models are still improving at or near epoch 100, the current 100-epoch budget is probably too short to fully judge this regime on EC50. However, the thresholded held-out results already show that the conservative regime harms the two smaller models much more than the large model
- Tuned EC50 comparison takeaway within experiment_002:
	- unlike EC50 experiment_001, this run has a single clear winner: large_mlp is strongest on every recorded held-out metric, including test loss, accuracy, precision, recall, F1, ROC-AUC, balanced accuracy, and total error count
	- the confusion matrices show that large_mlp did not merely shift the precision/recall tradeoff; it improved both classes at once relative to the smaller models, reducing false positives and false negatives simultaneously versus recommended_mlp and especially versus small_mlp
	- small_mlp became the weakest model by a wide margin, with the lowest test accuracy, recall, F1, and ROC-AUC plus the largest error count, 165 mistakes out of 861 test molecules
	- recommended_mlp stayed intermediate: its probability-quality metrics are much better than small_mlp, but it still lost clearly to large_mlp on held-out thresholded performance
- Direct comparison versus EC50 experiment_001:
	- small_mlp regressed heavily: test F1 fell from 0.8344 to 0.7815, test accuracy fell from 0.8548 to 0.8084, test ROC-AUC fell from 0.9288 to 0.8949, and test loss worsened from 0.3443 to 0.4082
	- recommended_mlp also regressed materially on thresholded metrics: test F1 fell from 0.8629 to 0.8182 and test accuracy fell from 0.8746 to 0.8374, although test loss improved slightly from 0.3686 to 0.3584, which suggests calibration improved somewhat while the final decision boundary became worse
	- large_mlp improved slightly but consistently: test F1 rose from 0.8543 to 0.8550, test accuracy rose from 0.8653 to 0.8676, test ROC-AUC rose from 0.9339 to 0.9340, and test loss improved from 0.3406 to 0.3266
	- the architecture ranking therefore changed again on EC50 under the tuned regime: recommended_mlp was the baseline winner in experiment_001, but large_mlp becomes the strongest model once dropout is increased and the learning rate is reduced
- Direct comparison versus IC50 experiment_002:
	- small_mlp transferred worse to EC50 than to IC50 under this conservative regime: test F1 fell from 0.8211 on IC50 to 0.7815 on EC50 and test ROC-AUC fell from 0.9241 to 0.8949
	- recommended_mlp is almost tied across the two tracks on the thesis primary metric, 0.8182 on EC50 versus 0.8168 on IC50, but EC50 shows slightly worse ranking quality and worse test loss
	- large_mlp transfers best across tracks: compared with IC50 experiment_002, the EC50 large_mlp improves slightly on every held-out metric, with test F1 rising from 0.8528 to 0.8550, test accuracy from 0.8626 to 0.8676, ROC-AUC from 0.9322 to 0.9340, and test loss improving from 0.3286 to 0.3266
	- these cross-track results strengthen the conclusion that this specific low-learning-rate, higher-dropout regime is mainly useful for the higher-capacity model
- Tuned EC50 run takeaway:
	- experiment_002 does not replace EC50 experiment_001; instead it establishes a second, more conservative comparison point under which the larger model becomes clearly dominant
	- the core lesson matches the earlier IC50 tuning study: the combination of higher dropout and 1e-4 learning rate reduces overfitting, but for the small and recommended models it becomes too conservative under a 100-epoch budget and materially harms held-out classification quality
	- for EC50 specifically, large_mlp is now the strongest tuned architecture under this regime, but the fact that all best checkpoints occur at epochs 98 through 100 suggests the next EC50 follow-up should not simply stop here. A higher learning rate or a longer budget is still justified if we want to know whether EC50 can reproduce the IC50 experiment_003-style recovery

# EC50 Follow-up Tuned Run: experiment_003

- Notebook used: deep_learning_pipeline.ipynb
- Follow-up tuned run executed and exported with these run metadata:
	- run directory = results/deep_learning_pipeline/EC50/experiment_003
	- run timestamp = 2026-05-03T00:01:16
	- dataset source = datasets/postprocessed-CHEMBL379_EC50/scaled_dataset/
	- device = cpu
	- torch version = 2.11.0+cpu
	- CUDA available = false
	- checkpoint policy = train the full 150 epochs, then restore the checkpoint with minimum validation loss
- Global notebook defaults used for this run:
	- dropout rate = 0.3
	- learning rate = 5e-4
	- weight decay = 5e-4
	- max epochs = 150
- Exported follow-up EC50 artifacts created:
	- results/deep_learning_pipeline/EC50/experiment_003/summary.csv
	- results/deep_learning_pipeline/EC50/experiment_003/summary.json
	- results/deep_learning_pipeline/EC50/experiment_003/test_confusion_summary.csv
	- results/deep_learning_pipeline/EC50/experiment_003/history/*.csv
	- results/deep_learning_pipeline/EC50/experiment_003/plots/*.png
- Follow-up EC50 results, architecture = 74 -> 32 -> 1 for small_mlp:
	- epochs trained = 150
	- best epoch = 148
	- validation loss = 0.3539
	- validation accuracy = 0.8537
	- validation precision = 0.8602
	- validation recall = 0.8122
	- validation F1 = 0.8355
	- validation ROC-AUC = 0.9241
	- test loss = 0.3516
	- test accuracy = 0.8525
	- test precision = 0.8598
	- test recall = 0.8096
	- test F1 = 0.8340
	- test ROC-AUC = 0.9254
	- test confusion matrix = TN 415, FP 52, FN 75, TP 319
- Follow-up EC50 results, architecture = 74 -> 64 -> 32 -> 1 for recommended_mlp:
	- epochs trained = 150
	- best epoch = 90
	- validation loss = 0.3331
	- validation accuracy = 0.8595
	- validation precision = 0.8438
	- validation recall = 0.8503
	- validation F1 = 0.8470
	- validation ROC-AUC = 0.9329
	- test loss = 0.3387
	- test accuracy = 0.8688
	- test precision = 0.8630
	- test recall = 0.8477
	- test F1 = 0.8553
	- test ROC-AUC = 0.9311
	- test confusion matrix = TN 414, FP 53, FN 60, TP 334
- Follow-up EC50 results, architecture = 74 -> 128 -> 64 -> 32 -> 1 for large_mlp:
	- epochs trained = 150
	- best epoch = 53
	- validation loss = 0.3265
	- validation accuracy = 0.8537
	- validation precision = 0.8401
	- validation recall = 0.8401
	- validation F1 = 0.8401
	- validation ROC-AUC = 0.9367
	- test loss = 0.3367
	- test accuracy = 0.8513
	- test precision = 0.8556
	- test recall = 0.8122
	- test F1 = 0.8333
	- test ROC-AUC = 0.9337
	- test confusion matrix = TN 413, FP 54, FN 74, TP 320
- Follow-up EC50 training-curve interpretation:
	- increasing the learning rate from 1e-4 to 5e-4 while keeping dropout and weight decay at 0.3 / 5e-4 materially recovered the two smaller models from the overly conservative EC50 experiment_002 regime
	- small_mlp no longer looks severely undertrained, but it still improves almost all the way to the end of the 150-epoch budget: its best validation-loss checkpoint is epoch 148, its best validation F1 is epoch 149, and the final validation-loss increase over the selected checkpoint is tiny. This means the 5e-4 regime is much healthier for small_mlp than experiment_002, although the current 150-epoch budget is still being used almost fully
	- recommended_mlp shows a better balance between recovery and stability: its validation-loss minimum occurs at epoch 90, while validation F1 stays near its peak into the later epochs and the final degradation is modest. This produces the strongest thresholded held-out performance within experiment_003 without the severe late-epoch collapse seen in some earlier large-model runs
	- large_mlp again overfits earliest in loss terms under the faster learning regime: validation loss bottoms at epoch 53 and then rises by roughly 0.035 by epoch 150, while validation F1 stays on a flatter plateau for much longer. This is the familiar pattern where ranking quality remains strong but probability calibration drifts as the extra epochs continue
	- overall, the EC50 experiment_003 curves mirror the IC50 experiment_003 lesson closely: the 5e-4 learning rate recovers useful learning relative to experiment_002, but the 150-epoch budget creates a long tail that is unnecessary for the larger model and only partially useful for the smaller ones
- Follow-up EC50 comparison takeaway within experiment_003:
	- recommended_mlp is the strongest model on the thesis primary metric, test F1, and also on test accuracy, recall-adjusted balance, and total error count; it made 113 total mistakes, fewer than small_mlp at 127 and large_mlp at 128
	- small_mlp remains the most conservative classifier in this run: it has the highest specificity, 0.8887, and the fewest false positives, but its recall stays clearly below recommended_mlp, so it gives up too many active compounds to win on F1
	- large_mlp retains the strongest ranking-oriented profile inside experiment_003, with the best validation ROC-AUC and nearly the best test ROC-AUC plus the best test loss, but it no longer converts that into the best fixed-threshold classification behavior. On the thesis metric, it finishes last of the three models by a small margin
	- this run therefore reinforces the same metric-policy point already seen in earlier experiments: the model with the best ROC-AUC or loss is not automatically the model with the best final thresholded classification decisions
- Direct comparison versus EC50 experiment_002:
	- small_mlp recovered strongly: test F1 rose from 0.7815 to 0.8340, test accuracy rose from 0.8084 to 0.8525, test ROC-AUC rose from 0.8949 to 0.9254, and test loss improved from 0.4082 to 0.3516
	- recommended_mlp also recovered materially: test F1 rose from 0.8182 to 0.8553, test accuracy rose from 0.8374 to 0.8688, test ROC-AUC rose from 0.9201 to 0.9311, and test loss improved from 0.3584 to 0.3387
	- large_mlp moved in the opposite direction: test F1 fell from 0.8550 to 0.8333, test accuracy fell from 0.8676 to 0.8513, and test loss worsened slightly from 0.3266 to 0.3367, while ROC-AUC stayed almost unchanged
	- these results confirm that the higher learning rate of 5e-4 successfully repaired the underfitting problem for small_mlp and recommended_mlp, but it removed the advantage that large_mlp had held under the slower experiment_002 regime
- Direct comparison versus EC50 experiment_001:
	- small_mlp almost exactly recovered its original EC50 baseline thresholded performance, but did not surpass it: test F1 changed only from 0.8344 to 0.8340, accuracy slipped slightly, and ROC-AUC also remained slightly lower
	- recommended_mlp came very close to its EC50 baseline winner run but still remained a bit below it on the thesis primary metric: test F1 fell from 0.8629 in experiment_001 to 0.8553 here and test accuracy fell from 0.8746 to 0.8688, although test loss improved substantially from 0.3686 to 0.3387 and test ROC-AUC improved from 0.9280 to 0.9311
	- large_mlp regressed meaningfully on thresholded metrics relative to the EC50 baseline run: test F1 fell from 0.8543 to 0.8333 and test accuracy fell from 0.8653 to 0.8513, while test ROC-AUC stayed effectively unchanged and test loss improved only slightly
	- the most important conclusion is that EC50 experiment_003 does not dethrone the original EC50 experiment_001 winner on the thesis primary metric. The best EC50 deep-learning test F1 recorded so far still belongs to experiment_001 recommended_mlp at 0.8629
- Direct comparison versus IC50 experiment_003:
	- small_mlp is slightly worse on EC50 than on IC50 under the same 5e-4 / 0.3 / 150-epoch regime: test F1 drops from 0.8426 to 0.8340 and test ROC-AUC drops from 0.9358 to 0.9254
	- recommended_mlp is materially better on EC50 than on IC50 for thresholded behavior under this regime: test F1 rises from 0.8351 on IC50 to 0.8553 on EC50 and test accuracy rises from 0.8483 to 0.8688, although ROC-AUC is slightly lower on EC50
	- large_mlp is materially worse on EC50 than on IC50 under this regime: test F1 falls from 0.8600 to 0.8333 and test accuracy falls from 0.8673 to 0.8513 even though ROC-AUC is slightly higher on EC50
	- the cross-track comparison again shows that architecture ranking is not stable across activity types: IC50 experiment_003 favored large_mlp, while EC50 experiment_003 favors recommended_mlp
- Follow-up EC50 run takeaway:
	- experiment_003 answers the follow-up question from EC50 experiment_002 positively for the two smaller models: moving from 1e-4 to 5e-4 recovers most of the lost performance and restores recommended_mlp as the best thresholded classifier in this regime
	- however, unlike IC50 experiment_003, the EC50 follow-up does not produce a new overall best run on the thesis primary metric. The original EC50 baseline experiment_001 remains slightly stronger than experiment_003 for recommended_mlp on test F1 and accuracy
	- therefore the most defensible current EC50 summary is: experiment_001 recommended_mlp remains the best locked EC50 deep-learning classifier by the thesis primary metric, experiment_002 large_mlp shows that the conservative regime can favor larger capacity, and experiment_003 demonstrates that the 5e-4 recovery rescues the smaller models but still does not clearly beat the EC50 baseline winner

# EC50 Midpoint-Dropout Follow-up Run: experiment_004

- Notebook used: deep_learning_pipeline.ipynb
- Follow-up tuned run executed and exported with these run metadata:
	- run directory = results/deep_learning_pipeline/EC50/experiment_004
	- run timestamp = 2026-05-03T00:16:33
	- dataset source = datasets/postprocessed-CHEMBL379_EC50/scaled_dataset/
	- device = cpu
	- torch version = 2.11.0+cpu
	- CUDA available = false
	- checkpoint policy = train the full 100 epochs, then restore the checkpoint with minimum validation loss
- Global notebook defaults used for this run:
	- dropout rate = 0.25
	- learning rate = 5e-4
	- weight decay = 5e-4
	- max epochs = 100
- Exported follow-up EC50 artifacts created:
	- results/deep_learning_pipeline/EC50/experiment_004/summary.csv
	- results/deep_learning_pipeline/EC50/experiment_004/summary.json
	- results/deep_learning_pipeline/EC50/experiment_004/test_confusion_summary.csv
	- results/deep_learning_pipeline/EC50/experiment_004/history/*.csv
	- results/deep_learning_pipeline/EC50/experiment_004/plots/*.png
- Follow-up EC50 results, architecture = 74 -> 32 -> 1 for small_mlp:
	- epochs trained = 100
	- best epoch = 94
	- validation loss = 0.3597
	- validation accuracy = 0.8490
	- validation precision = 0.8455
	- validation recall = 0.8198
	- validation F1 = 0.8325
	- validation ROC-AUC = 0.9197
	- test loss = 0.3489
	- test accuracy = 0.8479
	- test precision = 0.8525
	- test recall = 0.8071
	- test F1 = 0.8292
	- test ROC-AUC = 0.9243
	- test confusion matrix = TN 412, FP 55, FN 76, TP 318
- Follow-up EC50 results, architecture = 74 -> 64 -> 32 -> 1 for recommended_mlp:
	- epochs trained = 100
	- best epoch = 94
	- validation loss = 0.3335
	- validation accuracy = 0.8606
	- validation precision = 0.8374
	- validation recall = 0.8629
	- validation F1 = 0.8500
	- validation ROC-AUC = 0.9342
	- test loss = 0.3436
	- test accuracy = 0.8699
	- test precision = 0.8525
	- test recall = 0.8655
	- test F1 = 0.8589
	- test ROC-AUC = 0.9306
	- test confusion matrix = TN 408, FP 59, FN 53, TP 341
- Follow-up EC50 results, architecture = 74 -> 128 -> 64 -> 32 -> 1 for large_mlp:
	- epochs trained = 100
	- best epoch = 60
	- validation loss = 0.3230
	- validation accuracy = 0.8537
	- validation precision = 0.8252
	- validation recall = 0.8629
	- validation F1 = 0.8437
	- validation ROC-AUC = 0.9385
	- test loss = 0.3292
	- test accuracy = 0.8688
	- test precision = 0.8539
	- test recall = 0.8604
	- test F1 = 0.8571
	- test ROC-AUC = 0.9357
	- test confusion matrix = TN 409, FP 58, FN 55, TP 339
- Follow-up EC50 training-curve interpretation:
	- lowering dropout from 0.30 to 0.25 while keeping the 5e-4 learning rate and 5e-4 weight decay produced a more competitive regime than EC50 experiment_003, but the effect was architecture-specific rather than uniformly beneficial
	- small_mlp improved almost until the end of the 100-epoch budget and showed only mild late overfitting: validation loss bottomed at epoch 94, the best validation F1 and ROC-AUC both appeared slightly earlier at epoch 91, and the final degradation from the selected checkpoint was very small. Even so, the held-out gains were limited and this model remained clearly behind the two deeper alternatives
	- recommended_mlp again showed the strongest balance between stability and thresholded performance: validation loss bottomed at epoch 94, validation F1 peaked earlier at epoch 85, and the late-epoch deterioration in validation F1 was small even though loss drifted upward more noticeably. This produced the strongest held-out F1 and accuracy inside experiment_004
	- large_mlp still overfit earliest in loss terms: validation loss bottomed at epoch 60 while the best validation F1 and ROC-AUC arrived much later, at epochs 85 and 86. That pattern again indicates calibration drift more than a collapse in discrimination, and in this specific EC50 run the model still converted that into very strong held-out performance
	- overall, EC50 appears to tolerate the lighter 0.25 dropout better than IC50 did: the larger model does not collapse under this regime and the recommended model remains highly competitive, so the architecture ranking stays tight instead of breaking sharply in favor of one model only
- Follow-up EC50 comparison takeaway within experiment_004:
	- recommended_mlp is the strongest model on the thesis primary metric, test F1, and also edges out the others on test accuracy and total error count, with 112 total mistakes versus 113 for large_mlp and 131 for small_mlp
	- large_mlp is effectively tied on thresholded performance and remains the strongest ranking-oriented model inside the run, with the best test ROC-AUC, 0.9357, and the best test loss, 0.3292. The gap to recommended_mlp on test F1 is extremely small, only about 0.0018
	- small_mlp remains the most conservative classifier: it has the fewest false positives and the highest specificity, but it still gives up too much recall to compete with the stronger two-layer and three-layer models
	- this run is another strong example of why the thesis metric policy matters: recommended_mlp wins on the fixed-threshold classifier metric, while large_mlp wins on ranking and probability-quality metrics
- Direct comparison versus EC50 experiment_003:
	- small_mlp regressed slightly: test F1 fell from 0.8340 to 0.8292 and test accuracy fell from 0.8525 to 0.8479, while test loss improved slightly and ROC-AUC stayed almost flat
	- recommended_mlp improved slightly on the thesis primary metric: test F1 rose from 0.8553 to 0.8589 and test accuracy rose from 0.8688 to 0.8699, although test ROC-AUC decreased slightly and test loss worsened modestly
	- large_mlp improved materially: test F1 rose from 0.8333 to 0.8571, test accuracy rose from 0.8513 to 0.8688, test ROC-AUC rose from 0.9337 to 0.9357, and test loss improved from 0.3367 to 0.3292
	- these deltas show that reducing dropout from 0.30 to 0.25 helped large_mlp substantially on EC50 and gave recommended_mlp a small thresholded-performance lift, while small_mlp did not benefit
- Direct comparison versus EC50 experiment_002:
	- small_mlp and recommended_mlp both remain much better than in the overly conservative 1e-4 regime: test F1 rises from 0.7815 to 0.8292 for small_mlp and from 0.8182 to 0.8589 for recommended_mlp, with large gains in accuracy and ROC-AUC for both
	- large_mlp improves slightly even over its strong experiment_002 result: test F1 rises from 0.8550 to 0.8571, accuracy from 0.8676 to 0.8688, and ROC-AUC from 0.9340 to 0.9357
	- experiment_004 therefore dominates experiment_002 as the more useful 100-epoch EC50 follow-up: it preserves the large-model strength while avoiding the severe undertraining of the two smaller models
- Direct comparison versus EC50 experiment_001:
	- small_mlp remains below its original EC50 baseline: test F1 falls from 0.8344 to 0.8292 and test accuracy falls from 0.8548 to 0.8479
	- recommended_mlp comes very close to the original EC50 baseline winner but still does not beat it on the thesis primary metric: test F1 falls from 0.8629 to 0.8589 and accuracy falls from 0.8746 to 0.8699, although test loss improves materially from 0.3686 to 0.3436 and ROC-AUC improves from 0.9280 to 0.9306
	- large_mlp now slightly exceeds its EC50 baseline on thresholded and ranking metrics simultaneously: test F1 rises from 0.8543 to 0.8571, test accuracy from 0.8653 to 0.8688, test ROC-AUC from 0.9339 to 0.9357, and test loss improves from 0.3406 to 0.3292
	- the overall EC50 leaderboard on the thesis primary metric still does not change: experiment_001 recommended_mlp remains the strongest locked EC50 deep-learning result at test F1 = 0.8629, but experiment_004 recommended_mlp becomes the closest follow-up and experiment_004 large_mlp becomes the best EC50 large-model run so far
- Direct comparison versus IC50 experiment_004:
	- small_mlp is worse on EC50 than on IC50 under the same 0.25 / 5e-4 / 100-epoch regime: test F1 falls from 0.8469 to 0.8292 and ROC-AUC falls from 0.9385 to 0.9243
	- recommended_mlp is slightly better on EC50 than on IC50 for thresholded behavior under this regime: test F1 rises from 0.8571 to 0.8589 and test accuracy rises from 0.8673 to 0.8699, although ROC-AUC is lower and test loss is worse on EC50
	- large_mlp is materially better on EC50 than on IC50 under this same regime: test F1 rises from 0.8384 to 0.8571, test accuracy rises from 0.8483 to 0.8688, ROC-AUC rises from 0.9244 to 0.9357, and test loss improves from 0.3448 to 0.3292
	- the cross-track comparison therefore differs from the IC50 story: on IC50, dropout 0.25 hurt large_mlp substantially, but on EC50 the same lighter-dropout regime is actually one of the strongest large-model settings seen so far
- Cross-check against the EC50 optimized-large probe:
	- the standard deep-learning large_mlp in experiment_004 slightly exceeds the specialized optimized-large EC50 probe on test F1, 0.8571 versus 0.8541, and on ROC-AUC, 0.9357 versus 0.9333, while also improving test loss slightly
	- however, the optimized-large probe still keeps a small edge on test accuracy, so these two large-model results should be treated as effectively comparable rather than decisively separated
- Follow-up EC50 run takeaway:
	- experiment_004 confirms that the 0.25-dropout regime is a viable EC50 setting for the standard deep-learning notebook and is much healthier on EC50 than the analogous regime was on IC50
	- within this run, recommended_mlp is the best thresholded classifier and large_mlp is the best ranking-oriented model, with only a tiny gap between them on the thesis primary metric
	- even so, experiment_004 still does not dethrone the original EC50 baseline winner. The best locked EC50 deep-learning result on the thesis primary metric remains experiment_001 recommended_mlp at test F1 = 0.8629

# EC50 Classical ML Baseline Run: experiment_001

- Notebook used: classical_ml_baseline_pipeline.ipynb
- Run executed and exported with these run metadata:
	- run directory = results/classical_ml_baseline/EC50/experiment_001
	- run timestamp = 2026-05-03T00:38:38
	- dataset source = datasets/postprocessed-CHEMBL379_EC50/scaled_dataset/
	- decision threshold = 0.5
	- selection protocol = train on the training split only, select by validation F1, use validation ROC-AUC as the main secondary metric, and evaluate the frozen winner once on test
	- model-grid sizes = Logistic Regression 6, Random Forest 24, SVM 24
- Exported baseline artifacts created:
	- results/classical_ml_baseline/EC50/experiment_001/validation_trials.csv
	- results/classical_ml_baseline/EC50/experiment_001/family_winners.csv
	- results/classical_ml_baseline/EC50/experiment_001/selected_baseline_summary.csv
	- results/classical_ml_baseline/EC50/experiment_001/test_confusion_summary.csv
	- results/classical_ml_baseline/EC50/experiment_001/summary.json
	- results/classical_ml_baseline/EC50/experiment_001/plots/selected_baseline_test_confusion_matrix.png
- Validation-best family winners:
	- Random Forest winner: class_weight = balanced, n_estimators = 500, max_depth = None, min_samples_leaf = 3, validation accuracy = 0.8688, validation precision = 0.8593, validation recall = 0.8528, validation F1 = 0.8561, validation ROC-AUC = 0.9365, validation log loss = 0.3331
	- SVM winner: kernel = rbf, C = 10.0, gamma = 0.01, class_weight = balanced, validation F1 = 0.8474, validation ROC-AUC = 0.9272, validation log loss = 0.3450
	- Logistic Regression winner: C = 1.0, class_weight = balanced, validation F1 = 0.7589, validation ROC-AUC = 0.8554, validation log loss = 0.4746
- Validation ranking interpretation:
	- Random Forest clearly dominated the EC50 sweep at the family level: the top ten validation trials by F1 all came from Random Forest variants, and the selected 500-tree balanced model matched the best validation F1 while also holding the strongest ROC-AUC and log-loss combination among the tied top settings
	- the best SVM remained competitive but still trailed the selected Random Forest by about 0.0086 validation F1 and 0.0093 validation ROC-AUC
	- Logistic Regression underfit this descriptor space and never approached the non-linear models on the selection metric
- Frozen official EC50 classical baseline selected from validation only:
	- model family = Random Forest
	- hyperparameters = class_weight balanced, n_estimators 500, max_depth None, min_samples_leaf 3
- Frozen baseline test-set results:
	- test accuracy = 0.8780
	- test precision = 0.8833
	- test recall = 0.8452
	- test F1 = 0.8638
	- test ROC-AUC = 0.9445
	- test log loss = 0.3207
	- test confusion matrix = TN 423, FP 44, FN 61, TP 333
- EC50 confusion-matrix interpretation:
	- the held-out error profile is well balanced overall: specificity = 0.9058, sensitivity = 0.8452, balanced accuracy = 0.8755
	- false negatives slightly outnumber false positives, 61 versus 44, so this Random Forest is a bit more conservative on active calls than the best EC50 MLPs
	- even with that recall trade-off, the model keeps both high precision and high specificity, which is what lets it finish with the strongest overall EC50 accuracy and F1 among the locked runs so far
- Direct comparison versus the current EC50 deep-learning runs:
	- unlike the IC50 story, the EC50 classical baseline is not behind the best MLP checkpoints on the thesis primary metric; it slightly surpasses the current best EC50 deep-learning test F1, 0.8638 versus 0.8629 for experiment_001 recommended_mlp
	- the advantage is clearer on the main secondary metric: test ROC-AUC = 0.9445 for the Random Forest baseline versus 0.9357 for the strongest EC50 deep-learning ROC-AUC so far, experiment_004 large_mlp
	- the Random Forest baseline also currently has the strongest recorded EC50 test accuracy and precision, while the best EC50 MLP checkpoints still keep a recall edge because recommended_mlp in experiment_004 reaches 0.8655 recall and experiment_001 recommended_mlp reaches 0.8629 recall
	- the most defensible current EC50 conclusion is therefore stronger than on IC50: for this EC50 descriptor track, the frozen Random Forest baseline is presently the overall leader, with deep learning remaining highly competitive but not yet better on either the thesis primary metric or the main secondary metric
- Direct comparison versus the IC50 classical baseline:
	- both tracks selected Random Forest as the official classical winner, but the preferred hyperparameters differ: IC50 chose a shallower 200-tree model with max_depth = 10 and min_samples_leaf = 1, while EC50 chose a larger 500-tree ensemble with no depth cap and min_samples_leaf = 3
	- EC50 improves over IC50 on almost every aggregate held-out metric: test accuracy rises from 0.8531 to 0.8780, test precision from 0.7928 to 0.8833, test F1 from 0.8502 to 0.8638, test ROC-AUC from 0.9398 to 0.9445, and test log loss improves from 0.3376 to 0.3207
	- the one metric that moves in the opposite direction is recall: IC50 had 0.9167 test recall versus 0.8452 on EC50, so the EC50 Random Forest is much less aggressive about calling compounds active
	- this cross-track shift is useful for the thesis discussion: IC50 and EC50 are not just rescaled copies of the same problem. Even the same model family learns a different operating point, with IC50 favoring sensitivity and EC50 favoring a more precision-balanced classifier
- Parallel track takeaway after adding EC50 classical ML:
	- IC50 still supports the narrative that the best MLP checkpoints hold a small edge on thresholded classification while Random Forest leads on ranking quality
	- EC50 now shows the opposite headline: the frozen Random Forest baseline slightly leads even the best recorded MLP on test F1 and more clearly leads on ROC-AUC, accuracy, and precision
	- the combined evidence across both tracks strengthens the thesis argument that model family ranking depends on the bioactivity readout and should be reported empirically rather than assumed

Current todos:

- Analyze duplicat SMILES groupes ✅
- Record data curation decisions ✅
- Build labeled classification dataset ✅
- Regenerate descriptors from SMILES ✅
- Create train/validation/test split ✅
- Apply zero-variance filter ✅
- Apply correlation filter ✅
- Scale features ✅
- Wire notebook data pipeline ✅
- Add notebook DataLoader cell ✅
- Add notebook model-definition cell ✅
- Add notebook training utilities and experiment cells ✅
- Run first three-MLP DNN classification comparison ✅
- Summarize baseline results and add notebook visualizations ✅
- Define primary model-selection metric for the thesis ✅
- Implement reproducible classical ML baseline pipeline ✅
- Run Logistic Regression, Random Forest, and SVM validation sweep ✅
- Freeze the official classical baseline and evaluate it once on test ✅
- Compare the frozen classical baseline against the current deep-learning runs and record the conclusion ✅
- Define regression dataset rule
- Define the EC50 labeling policy exactly, mirroring the IC50 censor-aware rules ✅
- Build the EC50 classification curation dataset ✅
- Build the remaining EC50 curation and preprocessing stages with the same structure as the IC50 track ✅
- Run a tuned EC50 deep-learning follow-up and compare it against the EC50 baseline ✅
- Run the midpoint-dropout EC50 follow-up and compare it against the earlier EC50 runs ✅
- Point the optimized large-MLP notebook at the EC50 scaled dataset for the first parallel probe ✅
- Run the optimized large-MLP notebook first on the EC50 track as a quick-gain probe before the broader EC50 baseline study ✅
- Run the official deep-learning notebook on the EC50 track and record the first baseline comparison ✅
- Run the classical ML baseline on EC50 and compare the parallel IC50 versus EC50 tracks ✅
s