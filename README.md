# How to See Hidden Patterns in Metamaterials with Interpretable Machine Learning

This repository contains the code for experiments in the following paper

> Zhi Chen, Alexander Ogren, Chiara Daraio, L. Catherine Brinson, Cynthia Rudin
>
> [How to See Hidden Patterns in Metamaterials with Interpretable Machine Learning](https://www.sciencedirect.com/science/article/pii/S2352431622001717)
> 
> Extreme Mechanics Letters. https://doi.org/10.1016/j.eml.2022.101895

arXiv [pre-publication version](https://arxiv.org/abs/2111.05949)

## Data
Please download *bandgap_data.mat*, containing the 10 by 10 metamaterials unit-cells and their simulated (by FEA) dispersion curves, from [this google drive folder][https://drive.google.com/drive/folders/1gT5heJwiWtWyyMaE_JM9W7QCu1dwj-lX?usp=sharing]. After the dataset is downloaded, please put them in the */data* folder

## Code Details
### Structure-to-property Prediction
Run 
```
python3 compare_bacc.py
```
to compare the test balanced-accuracies of different ML models trained on raw features and shape-frequency features for predicting bandgap existence.
### Train sparse decision trees on shape frequency features
Run
```
python3 create_bin_datasets.py
```
to create binarized datasets to be used by GOSDT. Then run
```
python3 run_gosdt_sff.py
```
to get the sparse decision trees trained via [GOSDT algorithm][https://arxiv.org/abs/2006.08690].
### Train Unit-cell Template Sets
The unit-cell template algorithm contains two steps (1) preselection; (2) MIP optimization 
Run
```
python3 preselect_templates.py
```
to preselect a useful set of templates. Then run
```
python3 choose_templates.py
```
to get the optimal set of templates using MIP. Note that, we use [CPLEX optimizer][https://www.ibm.com/analytics/cplex-optimizer] to solve the MIP, please install it before running *choose_templates.py*.