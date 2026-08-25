"""
Usage example: predict optimum bitumen content (OBC) for a new Marshall
specimen using the pre-test (compaction-free) feature set — no parameter
measured DURING or AFTER Marshall compaction (air voids, VMA, VFA, Gmb,
stability, flow) is used, only properties available before any specimen
is compacted: aggregate gradation, aggregate quality indices, bitumen
properties, and pavement layer type.

The bundled model, held_out_test_model_pretest_stratified.pkl, was trained
on the 164-specimen training split only; the 42-specimen test split was
never seen during fitting. Its predictions on that test split reproduce
the manuscript's Table 3 metrics exactly (R^2 = 0.926, RMSE = 0.093,
MAE = 0.064) — this is the model referenced throughout the manuscript.

Usage:
    python predict_example_pretest.py
"""
import pickle

import pandas as pd

MODEL_FILE = "held_out_test_model_pretest_stratified.pkl"

with open(MODEL_FILE, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
feature_names = saved["feature_names"]
print("Model file:", MODEL_FILE)
print("Algorithm:", saved["best_model_name"])
print("Expected features:")
print(feature_names)

# Replace with your own material test results.
new_specimen = pd.DataFrame([{
    "layer_type": "binder_course",
    "passing_25_4mm": 100.0,
    "passing_19_1mm": 90.0,
    "passing_12_7mm": 70.5,
    "passing_9_5mm": 60.0,
    "passing_4_75mm": 43.0,
    "passing_2_0mm": 28.5,
    "passing_0_425mm": 12.0,
    "passing_0_177mm": 7.8,
    "passing_0_075mm": 4.8,
    "bitumen_specific_gravity": 1.033,
    "penetration": 57,
    "softening_point": 50,
    "los_angeles_loss": 24.0,
    "mgso4_loss": 6.0,
    "flakiness_index": 16.5,
}])[feature_names]

prediction = model.predict(new_specimen)
print(f"\nPredicted optimum bitumen content (pre-test estimate): {prediction[0]:.2f} %")
print("Note: this is a pre-compaction estimate. Laboratory Marshall verification")
print("is recommended before a final mix-design decision.")
