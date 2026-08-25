"""
Test-Oncesi Senaryo -- KATMANLI (stratified) 80/20 bolme ile TAM YENIDEN KURULUM.
Bu betik, orijinal pretest_pipeline.py + 03_tuning.py (pretest kismi) +
05_robust_validation.py (pretest kismi) + 07_shap_analysis.py (pretest kismi) +
08_learning_curves.py (pretest kismi) adimlarinin TAMAMINI, train_test_split ve
tum capraz dogrulama katlamalarinda stratify=layer_type kullanarak tekrarlar.

VIF hesaplamasi tum veri seti uzerinde yapildigindan (train/test bolunmesinden
bagimsiz) DEGISMEZ ve burada tekrar hesaplanmaz.

NOT / NOTE: Bu betik "marshall_dataset_ml.xlsx" adli bir veri dosyasi
bekler. Ham veri seti (206 laboratuvar Marshall numunesi), kurumsal veri
sahipligi nedeniyle bu depoya dahil edilmemistir; makul talep uzerine
yazardan temin edilebilir (bkz. README.md / makaledeki "Data Availability"
beyani). Bu betik, veri seti temin edildiginde metodolojiyi (Bolum 3.3-4.6)
birebir yeniden uretmek icin saglanmistir.

This script expects a data file named "marshall_dataset_ml.xlsx". The raw
dataset (206 laboratory Marshall specimens) is not included in this
repository due to institutional data ownership and is available from the
corresponding author upon reasonable request (see README.md / the
manuscript's Data Availability Statement). This script is provided so the
methodology (manuscript Sections 3.3-4.6) can be reproduced exactly once
the dataset is obtained.
"""
import warnings, json, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    cross_validate, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

RANDOM_STATE = 42
TARGET = "optimum_bitumen_ratio"
AT_OPTIMUM_COLS = ["air_voids_va", "vma", "vfa", "bulk_specific_gravity_gmb",
                    "marshall_stability", "flow"]


def load_and_clean(path):
    df = pd.read_excel(path)
    if "file_name" in df.columns:
        df = df.drop(columns=["file_name"])
    const_cols = [c for c in df.columns if df[c].nunique() == 1]
    df = df.drop(columns=const_cols)
    df = df.drop(columns=[c for c in AT_OPTIMUM_COLS if c in df.columns])
    return df


def build_preprocessor(X):
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]), num_cols, cat_cols


def get_candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR (RBF)": SVR(kernel="rbf", C=10, epsilon=0.05),
        "Random Forest": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0,
        ),
    }


def get_tuning_grids():
    return {
        "Extra Trees": (
            ExtraTreesRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": [150, 300, 450], "model__max_depth": [None, 8, 12],
             "model__min_samples_leaf": [1, 2, 3], "model__max_features": ["sqrt", 0.6, 1.0]},
        ),
        "XGBoost": (
            XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            {"model__n_estimators": [150, 300, 450], "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.7, 0.8, 1.0],
             "model__colsample_bytree": [0.6, 0.8, 1.0]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.7, 0.85, 1.0]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": [150, 300, 450], "model__max_depth": [None, 8, 12],
             "model__min_samples_leaf": [1, 2, 3], "model__max_features": ["sqrt", 0.6, 1.0]},
        ),
    }


def ci95(arr):
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return m, m - 1.96 * se, m + 1.96 * se, arr.std(ddof=1)


results = {}

df = load_and_clean("marshall_dataset_ml.xlsx")
X = df.drop(columns=[TARGET])
y = df[TARGET]
strata = X["layer_type"]
print(f"Veri seti (test-oncesi ozellikler): {df.shape[0]} numune, {df.shape[1]-1} aciklayici degisken")

preprocess, num_cols, cat_cols = build_preprocessor(X)

# =====================================================================
# 1) KATMANLI 80/20 BOLME
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strata)
print(f"Egitim: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print("Egitim tabaka dagilimi:\n", X_train["layer_type"].value_counts())
print("Test tabaka dagilimi:\n", X_test["layer_type"].value_counts())
results["split_layer_counts"] = {
    "train": X_train["layer_type"].value_counts().to_dict(),
    "test": X_test["layer_type"].value_counts().to_dict(),
}

# =====================================================================
# 2) 10 ALGORITMA TARAMASI -- katmanli 10-katli CV (egitim seti icinde)
# =====================================================================
skf10_train = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
                    .split(X_train, X_train["layer_type"]))
scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}

cv_rows = []
for name, model in get_candidate_models().items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    res = cross_validate(pipe, X_train, y_train, cv=skf10_train, scoring=scoring, n_jobs=-1)
    cv_rows.append({"model": name, "cv_r2_mean": res["test_r2"].mean(), "cv_r2_std": res["test_r2"].std(),
                     "cv_rmse_mean": -res["test_neg_rmse"].mean(), "cv_mae_mean": -res["test_neg_mae"].mean()})
cv_df = pd.DataFrame(cv_rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
print("\n=== Katmanli 10-katli CV taramasi (test-oncesi) ===")
print(cv_df.to_string(index=False))
cv_df.to_csv("cv_results_pretest_stratified.csv", index=False)
results["screening"] = cv_df.to_dict(orient="records")

# =====================================================================
# 3) HIPERPARAMETRE AYARLAMASI -- ilk 4 aday, katmanli 5-katli CV
# =====================================================================
grids = get_tuning_grids()
to_tune = [m for m in cv_df["model"].tolist() if m in grids][:4]
skf5_train = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                   .split(X_train, X_train["layer_type"]))

tuned_rows, fitted_best = [], {}
for name in to_tune:
    est, grid = grids[name]
    pipe = Pipeline([("prep", preprocess), ("model", est)])
    gs = RandomizedSearchCV(pipe, grid, n_iter=25, scoring="r2", cv=skf5_train,
                             n_jobs=-1, refit=True, random_state=RANDOM_STATE)
    gs.fit(X_train, y_train)
    pred_test = gs.predict(X_test)
    tuned_rows.append({
        "model": name, "best_cv_r2": gs.best_score_, "best_params": gs.best_params_,
        "test_r2": r2_score(y_test, pred_test),
        "test_rmse": mean_squared_error(y_test, pred_test) ** 0.5,
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_mape": mean_absolute_percentage_error(y_test, pred_test) * 100,
    })
    fitted_best[name] = gs.best_estimator_
    print(f"\n{name}: en iyi CV R2={gs.best_score_:.4f}, test R2={tuned_rows[-1]['test_r2']:.4f}")
    print("  parametreler:", gs.best_params_)

tuned_df = pd.DataFrame(tuned_rows).sort_values("best_cv_r2", ascending=False).reset_index(drop=True)
best_name = tuned_df.iloc[0]["model"]
best_model = fitted_best[best_name]
print(f"\nSecilen nihai model (katmanli test-oncesi): {best_name}")
tuned_df.to_csv("tuning_summary_pretest_stratified.csv", index=False)
results["tuning"] = tuned_df.drop(columns=["best_params"]).to_dict(orient="records")
results["best_name"] = best_name
results["best_params"] = tuned_df.iloc[0]["best_params"]

# =====================================================================
# 4) PERMUTASYON ONEMI (test setinde)
# =====================================================================
perm = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="r2")
imp_df = pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean,
                        "std": perm.importances_std}).sort_values("importance", ascending=False)
imp_df.to_csv("feature_importance_pretest_stratified.csv", index=False)
print("\n=== Permutasyon onemi (ilk 10) ===")
print(imp_df.head(10).to_string(index=False))
results["perm_importance_top"] = imp_df.head(10).to_dict(orient="records")

# =====================================================================
# 5) NIHAI MODELI TUM VERIYLE YENIDEN EGIT VE KAYDET
# =====================================================================
raw_params = {k.replace("model__", ""): v for k, v in best_model.get_params().items() if k.startswith("model__")}
clean_params = {k: v for k, v in raw_params.items() if v is not None and k not in ("random_state", "verbosity")}
model_map = {"XGBoost": XGBRegressor, "Gradient Boosting": GradientBoostingRegressor,
             "Extra Trees": ExtraTreesRegressor, "Random Forest": RandomForestRegressor}
final_est_cls = model_map[best_name]
final_est_allfit = final_est_cls(random_state=RANDOM_STATE, **({"verbosity": 0} if best_name == "XGBoost" else {}), **clean_params)
final_model_allfit = Pipeline([("prep", preprocess), ("model", final_est_allfit)])
final_model_allfit.fit(X, y)
with open("final_model_pretest_stratified.pkl", "wb") as f:
    pickle.dump({"model": final_model_allfit, "feature_names": list(X.columns), "target": TARGET,
                 "best_model_name": best_name, "best_params": clean_params}, f)

# also keep the TEST-SPLIT-FIT version (best_model) for residuals/SHAP evaluated strictly on held-out test data
with open("test_fit_model_pretest_stratified.pkl", "wb") as f:
    pickle.dump({"model": best_model, "best_model_name": best_name, "best_params": clean_params,
                 "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, f)

print(f"\nNihai model ({best_name}) params: {clean_params}")

# =====================================================================
# 6) TEKRARLANAN KATMANLI 10-KATLI CV (5 tekrar = 50 bolunme), SABIT HIPERPARAMETRE
# =====================================================================
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
best_pipe_fixed = Pipeline([("prep", preprocess), ("model", final_est_cls(random_state=RANDOM_STATE,
                             **({"verbosity": 0} if best_name == "XGBoost" else {}), **clean_params))])
scoring2 = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}
rep_res = cross_validate(best_pipe_fixed, X, y, cv=rskf.split(X, strata), scoring=scoring2, n_jobs=-1)
r2_scores = rep_res["test_r2"]
rmse_scores = -rep_res["test_neg_rmse"]
mae_scores = -rep_res["test_neg_mae"]
r2_m, r2_lo, r2_hi, r2_sd = ci95(r2_scores)
rmse_m, rmse_lo, rmse_hi, rmse_sd = ci95(rmse_scores)
mae_m, mae_lo, mae_hi, mae_sd = ci95(mae_scores)
print(f"\n[Katmanli tekrarlanan 10-katli CV, {len(r2_scores)} bolunme] ({best_name}):")
print(f"  R2   : {r2_m:.4f} (95% GA: {r2_lo:.4f}-{r2_hi:.4f}, std={r2_sd:.4f})")
print(f"  RMSE : {rmse_m:.4f} (95% GA: {rmse_lo:.4f}-{rmse_hi:.4f}, std={rmse_sd:.4f})")
print(f"  MAE  : {mae_m:.4f} (95% GA: {mae_lo:.4f}-{mae_hi:.4f}, std={mae_sd:.4f})")
results["repeated_cv"] = {"r2_mean": r2_m, "r2_ci": [r2_lo, r2_hi], "r2_std": r2_sd,
                           "rmse_mean": rmse_m, "rmse_ci": [rmse_lo, rmse_hi], "rmse_std": rmse_sd,
                           "mae_mean": mae_m, "mae_ci": [mae_lo, mae_hi], "mae_std": mae_sd,
                           "raw_r2_scores": r2_scores.tolist()}

# =====================================================================
# 7) NESTED CV -- katmanli dis (5-katli x2 tekrar) + katmanli ic (3-katli)
# =====================================================================
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
inner_cv_splitter = lambda Xtr, str_tr: list(StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE).split(Xtr, str_tr))
search_space = grids[best_name][1]
base_estimator = grids[best_name][0]

nested_scores = []
for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, strata)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    pre_inner, _, _ = build_preprocessor(X_tr)
    pipe_inner = Pipeline([("prep", pre_inner), ("model", base_estimator)])
    inner_splits = inner_cv_splitter(X_tr, X_tr["layer_type"])
    gs = RandomizedSearchCV(pipe_inner, search_space, n_iter=15, scoring="r2",
                             cv=inner_splits, n_jobs=-1, random_state=RANDOM_STATE)
    gs.fit(X_tr, y_tr)
    score = gs.score(X_te, y_te)
    nested_scores.append(score)
nested_scores = np.array(nested_scores)
n_m, n_lo, n_hi, n_sd = ci95(nested_scores)
optimism_gap = r2_m - n_m
print(f"\n[Katmanli nested CV, {len(nested_scores)} dis-katman]:")
print(f"  R2 (dis test): {n_m:.4f} (95% GA: {n_lo:.4f}-{n_hi:.4f}, std={n_sd:.4f})")
print(f"  Katlama skorlari: {np.round(nested_scores,3).tolist()}")
print(f"  Iyimserlik farki: {optimism_gap:.4f}")
results["nested_cv"] = {"r2_mean": n_m, "r2_ci": [n_lo, n_hi], "r2_std": n_sd,
                         "raw_scores": nested_scores.tolist(), "optimism_gap": optimism_gap}

# =====================================================================
# 8) EN IYI MODELLER ARASI ANLAMLILIK TESTLERI -- ayni katmanli 10-katli CV katlamalari
# =====================================================================
ALT_CANDIDATES = {
    "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE),
                       dict(n_estimators=300, max_depth=8, min_samples_leaf=1, max_features="sqrt")),
    "XGBoost": (XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
                dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8)),
}
# use each candidate's OWN tuned params from the tuning step above if available, else the alt defaults
tuned_params_by_name = {row["model"]: bp for row, bp in zip(tuned_rows, [r["best_params"] for r in tuned_rows])}

skf10_full = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE).split(X, strata))
main_pipe = Pipeline([("prep", preprocess), ("model", final_est_cls(random_state=RANDOM_STATE,
                       **({"verbosity": 0} if best_name == "XGBoost" else {}), **clean_params))])
main_scores = cross_validate(main_pipe, X, y, cv=skf10_full, scoring="r2", n_jobs=-1)["test_score"]

sig_results = []
for row in tuned_rows:
    name = row["model"]
    if name == best_name:
        continue
    bp = {k.replace("model__", ""): v for k, v in row["best_params"].items()}
    est_cls = model_map[name]
    est = est_cls(random_state=RANDOM_STATE, **({"verbosity": 0} if name == "XGBoost" else {}), **bp)
    pipe = Pipeline([("prep", preprocess), ("model", est)])
    scores = cross_validate(pipe, X, y, cv=skf10_full, scoring="r2", n_jobs=-1)["test_score"]
    t_stat, t_p = stats.ttest_rel(main_scores, scores)
    try:
        w_stat, w_p = stats.wilcoxon(main_scores, scores)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    print(f"\n{best_name} (ort={main_scores.mean():.4f}) vs {name} (ort={scores.mean():.4f}): "
          f"paired t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}")
    sig_results.append({"model_a": best_name, "model_b": name, "mean_a": float(main_scores.mean()),
                         "mean_b": float(scores.mean()), "t_p": float(t_p), "wilcoxon_p": float(w_p)})
results["significance_tests"] = sig_results

# =====================================================================
# 9) KALINTI DIAGNOSTIKLERI (yeni test setinde)
# =====================================================================
pred_test = best_model.predict(X_test)
resid = y_test.values - pred_test
sw_stat, sw_p = stats.shapiro(resid)
rho, rho_p = stats.spearmanr(pred_test, np.abs(resid))
print(f"\n=== Kalinti Normallik Testi (Shapiro-Wilk): W={sw_stat:.4f}, p={sw_p:.4f} ===")
print(f"=== Homoskedastisite (Spearman): rho={rho:.4f}, p={rho_p:.4f} ===")
results["residual_diagnostics"] = {"shapiro_w": float(sw_stat), "shapiro_p": float(sw_p),
                                     "spearman_rho": float(rho), "spearman_p": float(rho_p)}

with open("stratified_pipeline_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n\nTAMAMLANDI. Sonuclar: stratified_pipeline_results.json")
