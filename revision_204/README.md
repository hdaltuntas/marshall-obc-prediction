# Corrected 204-design analysis

This directory contains the corrected dataset, reproducible validation code,
saved numerical outputs and six figure exports. Python 3.12 was used.

## Run

```bash
cd revision_204
pip install -r requirements.txt
python revision_validation.py --input input_corrected_204.xlsx --output rerun --jobs 4
python annotate_binders.py --input input_corrected_204.xlsx --results rerun
python generate_figures.py --results rerun --figures rerun_figures
```

Full nested training takes longer than regenerating the figures from saved outputs:

```bash
python generate_figures.py --results results --figures regenerated_figures --reuse-diagnostics
```

## Files

- `input_corrected_204.xlsx`: analysis input with author-confirmed layer corrections dated 15 October 2026.
- `input_as_received.xlsx`: exact original upload retained for provenance; do not use it to reproduce the corrected results.
- `marshall_dataset_204.xlsx`: corrected refinery/type metadata with source provenance.
- `Supplementary_Data_S1_Revised_204.xlsx`: 204 records, fold assignments, metrics,
  audit and dictionary. Six post-test fields are retained for provenance and excluded
  from all pre-test model inputs.
- `Marshall_Validation_204.xlsx`: numerical analysis report.
- `results/`: individual held-out predictions, metrics, tuning trials, correlations,
  permutation diagnostics and normalized/source metadata.
- `figures/`: six 300 dpi PNG and vector PDF exports.
- `gradation_envelopes.json`: recorded-sieve limits transcribed from the supplied
  gradation table, not independently verified historical specifications.
- `SHA256SUMS.txt`: checksums for these files.

CSV outputs named `as_received` describe the input supplied to the current run
(`input_corrected_204.xlsx`); the original upload is preserved separately.

Use `input_corrected_204.xlsx` for both analysis scripts; the supplement has a
separate schema. Refinery, binder type/grade, quarry and dates are not predictors.
The original labels are decoded by `annotate_binders.py` according to the author's
final correction: 188 conventional binders from Aliağa, 15 PMB and one KMB from
Kırıkkale. All modified records are wearing-course designs.

## Population and validation

The original 206 source designs were reconciled by removing duplicate records
196/204/77 while retaining 193/199/76, correcting source 39 to binder course,
and adding wearing design 209 (208 was already present). The resulting population
is 204 designs: 62 base, 75 binder, 67 wearing, with 60 quarries and dates from
2011-07-21 to 2026-08-06. Source 207 passing_9_5mm=55.6 is retained as received.
Original and normalized labels remain separate for provenance. On 15 October 2026,
the author confirmed that source records 47, 51 and 57 are binder course, correcting
three base-course entry errors without changing measured values. All models and
validation partitions were recomputed with these corrected classes.

Primary validation uses five quarry-grouped outer folds and four grouped inner
folds with layer stratification. No quarry crosses training/test within those
splits. A record-wise nested reference is also reported. Temporal evaluation
trains on 162 designs from 2011–2021 and tests 42 from 2022–2026, with forward
whole-year inner tuning. Nine quarries occur in both temporal periods.
The cutoff was chosen using calendar-year counts before the results of this run.

Extra Trees was retained from the prior investigation; ten algorithms were not
reselected. Twenty-five random hyperparameter candidates are searched on
training-only MAE. All preprocessing is fitted within the training partition.
Seed 42. There are 16 raw pre-test predictors. Layer-mean baselines use only the
corresponding training records. Headline scores pool held-out predictions.

| Evaluation | Extra Trees R² | Extra Trees MAE | Layer-mean MAE |
|---|---:|---:|---:|
| Quarry-grouped nested | 0.906079 | 0.072691 | 0.069906 |
| Temporal test | 0.960310 | 0.052712 | 0.052878 |

Errors are percentage points. Target convention is percent by mass of dry
aggregate. Error sign is observed minus predicted. Within-layer grouped R² is
−0.131864 for base, −0.117946 for binder, and −0.034113 for wearing. High pooled
R² does not demonstrate strong within-layer prediction or consistent superiority
over layer means. Extra Trees has about 3.99% higher grouped MAE than layer means;
the temporal MAE difference is very small, while temporal RMSE favors layer means.

The three ablations and leave-one-quarry-out checks use previously reported fixed
parameters (150 trees, depth 8, leaf size 2, max_features sqrt); they are conditional
sensitivity analyses. Thirty held-out block permutations per grouped fold jointly
move each predictor block, across or within layers. These perturbation repeats
are not independent sampling replicates or causal effects. No naive fold-independent
confidence intervals or p-values are supplied. This is retrospective internal
validation, not prospective or external-laboratory validation. No final deployment
model has been fitted for the corrected population.

## Remaining data limitations

The author-confirmed binder labels resolve the former envelope discrepancies for
source 47/51/57; their passing_19_1mm values 90.6/91.5/90.3 are unchanged. All
204 records are compatible with a corresponding envelope over recorded sieves.
This is a consistency screen, not independent verification of historical reports
or certification of specification compliance. Missing 6 mm sieve measurements and subtype
IDs prevent certification of full specification compliance. Project, batch and
laboratory metadata remain unfilled. The actual OBC air-void targets, interpolation,
trial spacing, rounding and conditioning details still require source documentation.
The stored 0.05-point target grid does not establish experimental repeatability.
Changing workbook values does not refit models; rerun analyses if measurements change.
