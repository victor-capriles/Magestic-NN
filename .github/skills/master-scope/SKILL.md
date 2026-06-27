---
name: master-scope
description: Use this skill to understand the scope of the master's thesis project and its requirements. Additionally, this skill provides a detailed outline of the work to be done, including the objectives, methodology, and expected outcomes. It serves as a reference for the project's goals and the approach to be taken in developing and evaluating machine learning and deep learning models for the classification of compounds related to the Hepatitis C virus (HCV) target in ChEMBL. Also, it will be used as a the thesis Memoria, where we will store the work & results done in the project.
---

# Deep Learning for the Classification of Hepatitis C Virus-Related

## Introduction

Drug discovery is a complex and resource intensive process in which many candidate molecules must be evaluated before
identifying compounds with promising biological activity. Because experimental screening at a large scale requires considerable
time, cost and laboratory effort, computational methods have become useful support tools for prioritizing molecules during
earlier stages of research (Sadybekov & Katritch, 2023; Vamathevan et al.2019). In recent years, machine learning and deep learning
approaches have gained growing importance in this area because they can learn patterns from molecular data and assist inpredicting
whether a compound is likely to be active or not (Lenselink et al., 2017; Mayr et al., 2018).

In this context, this thesis explores the use of machine learning and deep learning methods for the classification of compounds
associated with the Hepatitis C virus (HCV) target in ChEMBL (Zdrazil et al., 2024). The study is based on a dataset filtered to
IC50 measurements, from which compounds are assigned a binary activity label. Using this dataset, a baseline machine learning model
and a deep learning model will be developed and compared under a reproducible experimental pipeline. The aim is to evaluate whether
the deep learning approach provides an advantage over a simpler baseline for this classification task.

## Context and justification of work

This thesis is situated at the intersection of machine learning and bioinformatics. The motivation comes from whether computational
models can effectively distinguish active from inactive compounds using public bioactivity data. This type of problem is relevant
because it provides a realistic setting in which predictive models can be evaluated as support tools for compound prioritization,
while remaining feasible within the scope of this thesis. Machine learning has become increasingly important in drug discovery for
exactly this type of decision-support setting, and HCV-related compound classification has also been explored through computational
modeling in prior work (Malik et al., 2021).

The work is framed around the HCV dataset available in ChEMBL, specifically the subset associated with target CHEMBL379 and filtered
to IC50 measurements. This dataset is suitable for the proposed study because it provides structured bioactivity values together with
molecular representations such as SMILES strings and derived descriptors. These characteristics make it possible to define a supervised
binary classification task and to construct a reproducible workflow for preprocessing, feature generation, model training, and evaluation
(Gaulton et al., 2012). From a methodological perspective, the project is justified by the opportunity to compare two different
modeling paradigms on the same problem. On the one hand, a classical machine learning model can serve as a baseline built from engineered
molecular descriptors. On the other hand, a deep learning model offers the possibility of learning more complex relationships from molecular
representations. Comparing both approaches is useful because it helps assess whether the additional complexity of deep learning is truly
beneficial for this dataset and task.

Therefore, the value of this thesis does not lie in producing a clinical decision system, but in developing and evaluating
(Mayr et al., 2018; Wu et al., 2018) a reproducible computational framework for activity classification using public ChEMBL bioactivity data.
The study is expected to provide a clearer view of the strengths and limitations of both machine learning and deep learning methods in
this context, while also establishing a solid experimental basis for future extensions of the work.

## Work objectives

### General objective

To develop and evaluate a reproducible machine learning and deep learning pipeline for the classification of compounds as active or
inactive using a ChEMBL-derived dataset associated with the HCV target (CHEMBL379) and filtered to IC50 measurements.

### Specific objectives

- To define the prediction task and construct a curated dataset from ChEMBL, including the selection of IC50 records, the application
of inclusion and exclusion criteria, and the derivation of a binary activity label.
- To preprocess the molecular data and generate suitable representations for modeling, including descriptor generation, feature
refinement, and dataset preparation for supervised learning.
- To define and justify a training, validation, and test strategy that allows a fair and reproducible evaluation of the models.
- To implement a baseline machine learning model and a deep learning model for the same classification task.
- To compare both approaches using consistent evaluation metrics and analyze their performance, limitations, and error patterns.
- To deliver a reproducible experimental workflow, together with the final thesis report presenting the methodology, results,
discussion, and future work.

## Impact on sustainability, ethics-social and diversity

While the work of this thesis doesn’t produce a clinical tool or a software solution, it can still have some impact in some areas
like sustainability. It can relate mainly to SDG 3 (Good Health and Well-being), since improving computational screening methods
can support drug discovery by helping prioritize certain compounds earlier in the pipeline also, it can relate to SDG 9 (Industry, Innovation
and Infrastructure) through the development of data-driven methods and reproducible software for biomedical research. However, in the
environmental aspect training deep learning models is quite expensive, it requires CPU or GPU usage, which requires compute and energy.

For the ethical and social impact, this work will avoid overstatement of the results. The models developed in this thesis are trained on
public bioactivity datasets and will be evaluated under controlled research settings. Therefore, the outcomes should be interpreted as a
computational benchmark rather than medical evidence.

Lastly, for diversity and inclusion, this work does not directly raise any fairness concerns related to individuals or populations.

## Focus and method followed

This thesis will be carried out as a supervised learning study focused on the classification of compounds as active or inactive
using public bioactivity data from ChEMBL. The methodological pipeline will begin with the selection and curation of compounds
associated with the HCV target CHEMBL379, restricted to IC50 measurements. From these values, a binary activity label will be derived
according to a predefined thresholding strategy. This step will allow the formulation of the prediction task as a binary classification
problem and will provide the basis for a consistent and reproducible modeling workflow.

After defining the dataset, the molecular information will be preprocessed in order to obtain suitable representations for model training.
The available compounds will be represented through molecular descriptors generated with RDKit, and the resulting feature space will be refined
by removing non-informative variables, such as descriptors with zero variance, as well as highly correlated features when appropriate.
Additional preprocessing steps, including normalization, will be applied when required by the selected models. These steps are intended to improve
data quality and ensure that both modeling approaches are trained on a clean and well-defined dataset.

From a modeling perspective, the study will compare two approaches. First, a classical machine learning model will be implemented as a baseline
using engineered molecular descriptors. This model will provide a reference point for the task and will help assess whether more complex
approaches offer a meaningful improvement. Second, a deep learning model will be developed for the same classification problem and evaluated
under the same experimental conditions. In this way, the thesis will not only measure predictive performance, but also examine whether the
additional complexity of deep learning is justified for this dataset. To ensure a fair comparison, the dataset will be divided into training,
validation, and test subsets following a fixed evaluation protocol.

Both the machine learning baseline and the deep learning model will be trained and assessed using the same data partitions and the same classification
metrics. In addition to overall predictive performance, the analysis will also consider model limitations and possible error patterns, with the
goal of identifying strengths and weaknesses in each approach. Finally, the methodological emphasis of the thesis will be on reproducibility.
All major decisions regarding dataset curation, label definition, preprocessing, feature selection, model training, and evaluation will be documented
clearly. The final outcome is expected to be a reproducible experimental pipeline that supports a transparent comparison between machine learning and
deep learning methods for compound activity classification using ChEMBL bioactivity data.

## Work schedule

Task Objective
T0 Deliver PEC1
T1 Define prediction task + dataset spec (ChEMBL IC50, units, label transform)
T2 Extract bioactivity data from ChEMBL
T3 Clean + standardize labels
T4 Define evaluation protocol & Dataset Freeze
T5 Train baseline models + write PEC2
T6 Implement deep model (pipeline)
T7 Train + tune deep model (logged experiments, controlled runs)
T8 Evaluate + compare (same splits/metrics) + error analysis + write PEC3
T9 Additional experiments
T10 Memoria
T11 Slides
T12 Defense preparation


## Brief summary products obtained

<Here we will sumarize later the products obtained in each task, such as the dataset specification, the curated dataset, the trained models,
the evaluation results, and the final thesis report.>

## Brief description of other memory chapters

<Here we will describe the content of other chapters in the thesis report, such as the introduction, methodology, results, discussion,
and future work sections.>

## State of the Art

### Bioactivity prediction in computational drug discovery

Bioactivity prediction has become one of the main applications of machine learning in computational drug discovery. The core idea is simple:
given molecular information, the goal is to predict whether a compound is likely to show relevant biological activity. Public databases
such as ChEMBL have made this line of work much more accessible because they provide standardized bioactivity records, assay information,
and target annotations. At the same time, the literature shows that success in this area depends not only on the model, but also on how
the data are curated, labeled, and evaluated.

A common way to formulate this problem is to transform continuous potency values such as IC50, EC50, Ki, or Kd into a binary classification task,
where compounds are labeled as active or inactive. This makes the prediction problem easier to handle and is widely used in QSAR and target
prediction studies. However, the threshold used to define activity is not a minor detail. It directly affects class balance and can strongly
influence how difficult the task becomes. Some studies use stricter thresholds to avoid having almost all compounds labeled as active, while
others remove borderline compounds altogether in order to reduce ambiguity near the decision boundary.

Another recurring theme in the literature is that bioactivity prediction is highly sensitive to data quality. Even when databases such as ChEMBL
standardize activity types and units, predictive performance can still be affected by duplicated records, inconsistent measurements, and
heterogeneous assay settings. This becomes even more important when the dataset is built around a broad organism-level target instead of a single
well-defined protein, since the biological context may be more variable and therefore harder to model consistently. For this reason, many studies
emphasize that careful curation is not just a preprocessing step, but a central part of the modeling process itself.

Classical machine learning methods remain a strong reference point in bioactivity prediction. In many studies, molecules are represented through
engineered descriptors or fingerprints, and models such as random forests, support vector machines, logistic regression, or gradient boosting are
trained on these fixed-length representations. These approaches continue to be widely used because they are efficient, relatively robust, and often
highly competitive. Large benchmark studies based on ChEMBL-style data show that strong classical baselines are still difficult to beat, even when
compared against more complex deep learning models (Lane et al., 2021; Mayr et al., 2018). This is important because it challenges a common assumption
in the field: deep learning is not automatically better just because it is more complex. In practice, classical methods remain part of the current
state of the art, especially when combined with good molecular representations such as ECFP fingerprints or physicochemical descriptors. For that reason,
they should not be treated as outdated baselines, but as necessary comparators in any serious bioactivity classification study (Rogers & Hahn, 2010).

For this thesis, that means the machine learning model is not included merely as a formality. It plays an essential role in showing whether the
deep learning model offers a real improvement or whether a simpler and more established approach is already sufficient for the CHEMBL379
classification task.

### Deep learning methods for bioactivity prediction

Deep learning has received increasing attention in bioactivity prediction because it offers the possibility of learning more complex patterns directly
from molecular data. This has led to several families of models in the literature, including feed-forward neural networks trained on descriptor vectors,
sequence models trained on SMILES strings, and graph neural networks that operate on molecular graphs. The general motivation behind these approaches
is to reduce dependence on manual feature design and allow the model to learn richer molecular representations.

Even so, current evidence suggests a more balanced view than the idea that deep learning consistently dominates classical methods. Some benchmark
studies report gains for deep learning, especially in multitask or large-scale settings, but others show little or no consistent advantage when
the comparison includes strong classical baselines and more realistic evaluation protocols. In fact, some comparative studies report that descriptor-based
approaches remain more accurate on average and are usually more computationally efficient than graph-based alternatives (Jiang et al., 2021). Because of that,
the key question is not whether deep learning is universally better, but when its added complexity is justified.

For a thesis like this one, that question is especially relevant. The value of the deep learning model lies in testing whether it can provide a meaningful advantage
on this dataset, not in assuming that it should outperform the baseline by default.

### Molecular representations used in the literature

One useful way to understand the literature is by looking at the molecular representations used as model input. A first group is based on engineered
descriptors and fingerprints. These are predefined numerical summaries of a molecule, such as physicochemical properties, topological descriptors,
and structural fingerprints. Their main advantage is that they convert each molecule into a fixed-size vector that can be used directly by
classical machine learning methods and also by feed-forward neural networks.

A second common representation is SMILES, which treats the molecule as a text sequence. In this setting, sequence-based deep learning models
attempt to learn relevant structural patterns directly from the molecular string. This approach is attractive because it reduces the need for manual feature
engineering, but it is also more sensitive to dataset size and model design (Mayr et al., 2018; Weininger, 1988).

A third representation is the molecular graph, where atoms are modeled as nodes and bonds as edges. Graph neural networks are appealing because
they preserve more of the native molecular structure and can, in principle, learn richer structural interactions. Even so, current benchmark results
suggest that graph-based models do not dominate all settings. Their strengths tend to appear more clearly in larger or multitask problems, while
descriptor-based pipelines remain very competitive in many standard bioactivity benchmarks.

### Evaluation strategies and methodological challenges

A repeated finding across the literature is that evaluation strategy can strongly affect the conclusions of a study. Random train-test splits are common,
but they can give optimistic estimates of performance because closely related molecules may appear in both sets. For this reason, several benchmark
studies recommend more demanding alternatives such as scaffold-based or temporal splits, which provide a more realistic picture of how well a model
generalizes to new chemistry or future data (Sheridan, 2013; Wu et al., 2018).

This is not a minor technical detail. In large ChEMBL-based benchmarks, performance can drop substantially when moving from random to temporal validation, even
when the relative ranking between methods remains informative. The same general concern appears in MoleculeNet-style benchmarks, where scaffold-based splits are
often discussed as a stronger test of chemical generalization (Landrum et al., 2023; Sheridan, 2013). Together, these results suggest that split choice is
one of the most important design decisions in a bioactivity prediction study.

The same applies to metric selection. Accuracy alone is often insufficient, especially when the classes are imbalanced. For this reason, the literature frequently
reports metrics such as ROC-AUC, PR-AUC, and MCC, and in some cases repeated runs or repeated splits are also used to estimate variability (Richardson et al., 2024).
These practices matter because they help avoid conclusions based on a single favorable partition or on a metric that hides imbalance effects. HCV-focused studies
and relation to this thesis Within the HCV literature, many predictive studies focus on specific viral proteins, especially NS5B polymerase, rather than on broader
organism-level targets. These studies are useful because they provide concrete examples of how to define activity thresholds, select descriptors, evaluate scaffold-aware
performance, and report class-sensitive metrics. For instance, one open-access NS5B study built a random forest classifier using explicit IC50 cutoffs and reported
an independent-test accuracy of 84.4% after descriptor selection (Wei et al., 2016). 

A more recent study, StackHCV, also focused on NS5B inhibitors and proposed an integrative machine-learning framework based on heterogeneous fingerprints and stacked models,
reinforcing that HCV-related compound classification remains an active area of research (Malik et al., 2021) At the same time, these studies should not be mapped
directly onto the present thesis. The dataset used here is associated with CHEMBL379, which corresponds to the broader Hepatitis C virus target rather than
to a single protein target such as NS5B. Because of that, NS5B-focused studies are more useful here as methodological references than as direct expectations
of model performance. What transfers best from that literature is not the exact reported accuracy, but the logic behind thresholding, validation, and reproducibility. Taken
together, the literature points to a clear gap that is relevant for this work. On one side, large benchmark studies show that deep learning is promising,
but not consistently superior across bioactivity datasets. On the other hand, HCV-focused studies show that threshold selection, molecular representation,
and validation strategy can strongly influence the final conclusions. What is less explored is how these lessons transfer to broader and potentially more
heterogeneous targets such as CHEMBL379, where the biological context is less specific than in single-protein studies such as those centered on NS5B.

For that reason, the contribution of this thesis is best framed as a careful and reproducible comparison between machine learning and deep learning methods for
binary classification using CHEMBL379 IC50 data. The goal is not to prove in advance that the more complex model is better, but to evaluate both approaches
under the same curation process, the same labeling strategy, and the same evaluation conditions.
