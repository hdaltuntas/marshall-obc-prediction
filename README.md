# Marshall OBC Prediction — Pre-Test Feature Set

Code accompanying the manuscript *"Prediction of Optimum Bitumen Content Using
Tree-Based Machine Learning and SHAP-Based Interpretability Prior to the
Marshall Test"*. Predicts optimum bitumen content (OBC) for Marshall mix
design using only properties available **before** any specimen is compacted:
aggregate gradation, aggregate quality indices, bitumen properties, and
pavement layer type.

## Repository contents

| File | Description |
|---|---|
| `pretest_pipeline_stratified.py` | Full analysis pipeline: preprocessing, 10-fold stratified model screening across 10 algorithms, hyperparameter tuning, repeated (50-split) and nested cross-validation, paired t-test / Wilcoxon significance testing, VIF and Shapiro–Wilk residual diagnostics. Reproduces manuscript Sections 3.3–4.6. |
| `predict_example_pretest.py` | Minimal usage example: load the trained model and predict OBC for one new specimen. |
| `held_out_test_model_pretest_stratified.pkl` | Trained Extra Trees pipeline, fit **only** on the 164-specimen training split; the 42-specimen test split was never seen during fitting. Its predictions reproduce the manuscript's Table 3 test-set metrics exactly (R² = 0.926, RMSE = 0.093, MAE = 0.064). This is the model referenced throughout the manuscript. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

## Usage

Run a scripted single-specimen prediction with the trained model:

```bash
python predict_example_pretest.py
```

Reproduce the full methodology and statistical validation reported in
Sections 3.3–4.6 (requires the raw dataset — see below):

```bash
python pretest_pipeline_stratified.py
```

## Data availability

The raw dataset (206 laboratory Marshall specimens, General Directorate of
Highways, 13th Regional Directorate) is not included in this repository due
to institutional data ownership. It is available from the corresponding
author upon reasonable request. The trained model file (`.pkl`) included
here allows prediction on new specimens and verification of the reported
test-set results without requiring the raw dataset.

## Citation

If you use this code, please cite the manuscript (details to be added upon
publication).

## License

MIT — see `LICENSE`. This applies to the code only; it does not extend to
the (unincluded) proprietary laboratory dataset.
