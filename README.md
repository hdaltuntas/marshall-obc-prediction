# Marshall OBC Prediction — Pre-Test Features

Estimate Marshall optimum bitumen content from measurements available before
specimen preparation: pavement layer, gradation, aggregate quality and binder
properties.

## Current corrected analysis

The [204-design analysis](revision_204/README.md) includes corrected datasets,
quarry-grouped nested and temporal validation, layer-mean comparisons, individual
held-out predictions, six figures and reproducible scripts.

```bash
cd revision_204
pip install -r requirements.txt
python revision_validation.py --input input_corrected_204.xlsx --output rerun --jobs 4
python annotate_binders.py --input input_corrected_204.xlsx --results rerun
python generate_figures.py --results rerun --figures rerun_figures
```

Python 3.12 was used. To regenerate figures from saved diagnostics, see the
[revision instructions](revision_204/README.md).

Quarry-grouped Extra Trees achieves R²=0.906079 and MAE=0.072691 percentage points,
versus layer-mean MAE=0.069906. Temporal MAE is 0.052712 versus baseline 0.052878.
Records 47, 51 and 57 were corrected to binder course after author confirmation
on 15 October 2026; measured values are unchanged and all analyses were rerun.
Within-layer performance is limited; pooled R² must not be interpreted as strong
within-layer prediction or consistent superiority. The revised evaluation is
retrospective internal validation. No updated deployment model is included.

## Earlier 206-record implementation

The existing root-level `pretest_pipeline_stratified.py`,
`predict_example_pretest.py`, `held_out_test_model_pretest_stratified.pkl` and
`requirements.txt` belong to the earlier 206-record analysis and are retained
for provenance. The serialized model has not been retrained on the corrected
204 designs. Its earlier scores and statistical tests do not describe the
current corrected analysis. Use the revision directory for current results.

## Data and license

The corrected input and supplementary data are included in `revision_204/`.
The MIT license applies to code; it does not transfer ownership of the
laboratory dataset. See [LICENSE](LICENSE).

## Citation

Publication citation details will be added when available.
