# HepaData: Liver Disease Prediction

An end-to-end machine learning study that predicts liver disease from routine blood test results, using the [Indian Liver Patient Dataset (ILPD)](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset). Course project for **BBF304E** at Istanbul Technical University.

The focus is on evaluation done carefully: separate tuning and evaluation splits, repeated cross-validation, statistical significance tests, and a final model chosen for what it actually gets right, not for its accuracy alone.

## Key result

The dataset is imbalanced (71% liver disease, 29% healthy), so accuracy is misleading. Gradient Boosting and Random Forest both reach **72.8% test accuracy**, but Gradient Boosting gets there by labelling almost every patient as sick: it identifies only 5 of 33 healthy patients. **Random Forest** was selected as the final model.

| Metric (held-out test set) | Gradient Boosting | **Random Forest** |
|---|---|---|
| Balanced accuracy | 0.557 | **0.737** |
| MCC | 0.203 | **0.435** |
| Specificity | 0.152 | **0.758** |
| Recall | **0.963** | 0.716 |
| ROC-AUC | 0.748 | **0.814** |
| Std. of balanced accuracy across 5 seeds | 0.048 | **0.040** |

A McNemar test (p = 0.880) shows both models make the same total number of errors on the test set; they differ in *which* errors they make. Gradient Boosting's higher recall is statistically significant (Wilcoxon, p = 0.002) but comes at the cost of near-zero specificity.

## Dataset

- 583 patients, 10 features (age, gender, bilirubin, liver enzymes, proteins) and a binary label
- 570 patients after removing 13 duplicate rows
- 4 missing values in `Albumin_and_Globulin_Ratio`, imputed with the median inside the pipeline
- Bilirubin and enzyme features are heavily right-skewed with many outliers

## Method

1. **Exploration** (`01_data_exploration.ipynb`): summary statistics, missing values, duplicates, class and feature distributions, IQR outlier analysis, correlations.
2. **Baselines** (`02_baseline_models.ipynb`): a majority-class Dummy Classifier and Logistic Regression, with all preprocessing fitted on the training split only.
3. **Model comparison** (`03_model_comparison.ipynb`): Logistic Regression, SVM, Random Forest, Gradient Boosting and an MLP.
   - Hyperparameters tuned with an inner stratified 5-fold CV, kept separate from evaluation
   - Models evaluated with 5×2 cross-validation and on an independent test set
   - Wilcoxon and McNemar tests for pairwise significance
4. **Deep analysis** (`04_ablation_feature_importance.ipynb`) of the two best candidates:
   - Class-weighted training and SMOTE oversampling
   - Feature-group ablation (demographics, bilirubin, enzymes, proteins)
   - Training-size experiment and learning curves
   - Impurity-based and permutation feature importance
   - Robustness over five random seeds
   - Misclassification analysis and the final model decision

## Repository structure

```text
├── data/
│   ├── raw/ilpd.csv
│   └── processed/ilpd_clean.csv
├── notebooks/        # 01–04, run in order
├── results/
│   ├── figures/      # all plots
│   ├── tables/       # all metric tables as CSV
│   └── models/       # tuned parameters and candidate models
├── src/              # data loading, preprocessing, models, evaluation, statistics, plots
└── requirements.txt
```

## Running it

```bash
git clone https://github.com/sadikerimi/BBF304E_2526_HepaData.git
cd BBF304E_2526_HepaData
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order (01 → 04); later notebooks read results saved by earlier ones.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, SciPy, Matplotlib, Jupyter
