import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed

FEATURES_CSV = r"./features_all_10s.csv"
OUT_DIR = r"./loso_lgb_modalities"

SUBJECT_COL = "part_NO"
TASK_COL = "task"
LABEL_COLS = [f"y{i}" for i in range(11)]

THRESH = 0.2
RANDOM_STATE = 42

# Recommended for AutoDL 32 vCPU: outer parallelism + small per-model threads
N_JOBS_OUTER = 8          # How many label models to train in parallel (per fold)
N_JOBS_PER_MODEL = 2      # Threads used inside each LightGBM model
# Rule of thumb: 8*2=16 threads is usually stable; you can also try 10*2 or 12*2 to increase CPU utilization

# Early stopping settings
EARLY_STOPPING_ROUNDS = 50
VALID_FRACTION_SUBJECTS = 0.2  # Sample 20% subjects from the training subjects as validation (subject-level to avoid leakage)
MIN_VALID_SUBJECTS = 1         # Ensure at least 1 subject for validation (fallback when too few training subjects)

MODALITY_SETS = {
    "EEG": ["eeg_"],
    "car": ["car_"],
    "emo": ["emo_"],
    "EEG+car": ["eeg_", "car_"],
    "EEG+emo": ["eeg_", "emo_"],
    "EEG+car+emo": ["eeg_", "car_", "emo_"],
}


def eval_f1(y_true, y_pred):
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_label_f1(y_true_2d, y_pred_2d, label_cols):
    out = {}
    for j, ycol in enumerate(label_cols):
        out[f"f1_{ycol}"] = f1_score(y_true_2d[:, j], y_pred_2d[:, j], zero_division=0)
    return out


def pick_feature_columns_by_prefix(df: pd.DataFrame, prefixes):
    forbidden = set([SUBJECT_COL, TASK_COL] + LABEL_COLS)
    cols = []
    for c in df.columns:
        if c in forbidden:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(c.startswith(p) for p in prefixes):
            cols.append(c)
    return cols


def _split_train_valid_by_subject(groups_tr, valid_fraction=0.2, seed=42):
    """Return a boolean mask for validation samples (within the training set), split by subject."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups_tr)
    n_valid_subj = int(round(len(uniq) * valid_fraction))
    n_valid_subj = max(MIN_VALID_SUBJECTS, min(len(uniq) - 1, n_valid_subj)) if len(uniq) >= 2 else 0
    if n_valid_subj <= 0:
        return None  # cannot split

    valid_subj = rng.choice(uniq, size=n_valid_subj, replace=False)
    is_valid = np.isin(groups_tr, valid_subj)
    # Ensure both parts are non-empty
    if is_valid.sum() == 0 or (~is_valid).sum() == 0:
        return None
    return is_valid


def _fit_one_label_with_es(X_tr, y_tr, groups_tr, X_te, base_params, seed_offset=0):
    """Train one binary label with early stopping (subject-level validation split)."""
    # Use a constant predictor if the training set contains only one class
    if np.unique(y_tr).size < 2:
        p = float(np.mean(y_tr))
        return np.full((len(X_te),), p, dtype=np.float32), {
            "status": "constant_train",
            "best_iter": 0,
            "train_pos_rate": float(np.mean(y_tr)),
        }

    is_valid = _split_train_valid_by_subject(
        groups_tr,
        valid_fraction=VALID_FRACTION_SUBJECTS,
        seed=RANDOM_STATE + seed_offset,
    )

    clf = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        **base_params,
        random_state=RANDOM_STATE + seed_offset,
    )

    if is_valid is None:
        # No valid split possible -> train without early stopping
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1].astype(np.float32)
        meta = {
            "status": "no_valid_split",
            "best_iter": int(getattr(clf, "best_iteration_", 0) or 0),
            "train_pos_rate": float(np.mean(y_tr)),
        }
        return prob, meta

    X_tr_in = X_tr[~is_valid]
    y_tr_in = y_tr[~is_valid]
    X_val = X_tr[is_valid]
    y_val = y_tr[is_valid]

    # If the validation split accidentally has only one class (rare), fall back to no early stopping
    if np.unique(y_val).size < 2 or np.unique(y_tr_in).size < 2:
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1].astype(np.float32)
        meta = {
            "status": "degenerate_valid",
            "best_iter": int(getattr(clf, "best_iteration_", 0) or 0),
            "train_pos_rate": float(np.mean(y_tr)),
        }
        return prob, meta

    clf.fit(
        X_tr_in, y_tr_in,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            # verbose=False: suppress per-iteration logs; change to verbose=10 if you want to monitor
            __import__("lightgbm").early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
        ],
    )
    best_iter = int(getattr(clf, "best_iteration_", 0) or 0)
    prob = clf.predict_proba(X_te)[:, 1].astype(np.float32)
    meta = {
        "status": "early_stopped",
        "best_iter": best_iter,
        "train_pos_rate": float(np.mean(y_tr)),
        "valid_size": int(len(y_val)),
    }
    return prob, meta


def loso_train_eval(df: pd.DataFrame, feat_cols, setting_name: str):
    X = df[feat_cols].to_numpy(dtype=np.float32)
    Y = df[LABEL_COLS].astype(int).to_numpy()
    groups = df[SUBJECT_COL].to_numpy()

    unique_subjects = np.unique(groups)
    gkf = GroupKFold(n_splits=len(unique_subjects))

    oof_prob = np.full((len(df), len(LABEL_COLS)), np.nan, dtype=np.float32)
    oof_pred = np.full((len(df), len(LABEL_COLS)), -1, dtype=np.int8)

    base_params = dict(
        n_estimators=5000,          # With early stopping, you can set a large upper bound
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        n_jobs=N_JOBS_PER_MODEL,
        verbosity=-1,          # Suppress most Info/Warning logs
        force_col_wise=True,   # Avoid the overhead/log message about auto-choosing col-wise
    )

    fold_reports = []
    fold_meta = []  # Store per-fold, per-label metadata such as best_iter

    t0 = time.time()
    n_folds = len(unique_subjects)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups=groups), start=1):
        fold_t0 = time.time()
        test_subject = int(np.unique(groups[te_idx])[0])

        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]
        groups_tr = groups[tr_idx]

        # Train 11 label models in parallel
        results = Parallel(n_jobs=N_JOBS_OUTER, prefer="processes")(
            delayed(_fit_one_label_with_es)(
                X_tr, Y_tr[:, j], groups_tr, X_te, base_params, seed_offset=fold * 100 + j
            )
            for j in range(len(LABEL_COLS))
        )

        prob = np.stack([r[0] for r in results], axis=1)  # (n_test, 11)
        meta_j = [r[1] for r in results]

        pred = (prob >= THRESH).astype(np.int8)

        oof_prob[te_idx] = prob
        oof_pred[te_idx] = pred

        rep = eval_f1(Y_te, pred)
        rep.update(per_label_f1(Y_te, pred, LABEL_COLS))
        rep.update({
            "fold": fold,
            "test_subject": test_subject,
            "n_test": int(len(te_idx)),
            "fold_seconds": float(time.time() - fold_t0),
        })
        fold_reports.append(rep)

        fold_meta.append({
            "fold": fold,
            "test_subject": test_subject,
            "label_meta": {LABEL_COLS[j]: meta_j[j] for j in range(len(LABEL_COLS))}
        })

        elapsed = time.time() - t0
        avg_fold = elapsed / fold
        eta = avg_fold * (n_folds - fold)
        print(
            f"[{setting_name}] Fold {fold}/{n_folds} subj={test_subject} "
            f"microF1={rep['micro_f1']:.4f} macroF1={rep['macro_f1']:.4f} "
            f"time={rep['fold_seconds']:.1f}s ETA={eta/60:.1f}m"
        )

    overall = eval_f1(Y, oof_pred)
    overall.update(per_label_f1(Y, oof_pred, LABEL_COLS))
    overall["total_seconds"] = float(time.time() - t0)

    return overall, fold_reports, fold_meta, oof_prob, oof_pred


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=LABEL_COLS).reset_index(drop=True)

    all_cols = df.columns.tolist()
    for pref in ["eeg_", "car_", "emo_"]:
        if not any(c.startswith(pref) for c in all_cols):
            print(f"[WARN] No columns with prefix '{pref}' found. Check your feature names.")

    summary_rows = []

    for setting_name, prefixes in MODALITY_SETS.items():
        feat_cols = pick_feature_columns_by_prefix(df, prefixes)
        if not feat_cols:
            print(f"[SKIP] {setting_name}: no feature columns found for prefixes={prefixes}")
            continue

        print(f"\n=== Running {setting_name} | n_features={len(feat_cols)} ===")
        t_set0 = time.time()

        overall, fold_reports, fold_meta, oof_prob, oof_pred = loso_train_eval(df, feat_cols, setting_name)

        set_sec = time.time() - t_set0
        print(
            f"[{setting_name}] OOF micro_F1={overall['micro_f1']:.4f}  "
            f"macro_F1={overall['macro_f1']:.4f}  time={set_sec/60:.1f}m"
        )

        setting_dir = os.path.join(OUT_DIR, setting_name.replace("+", "_"))
        os.makedirs(setting_dir, exist_ok=True)

        with open(os.path.join(setting_dir, "overall_report.json"), "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)

        with open(os.path.join(setting_dir, "fold_reports.json"), "w", encoding="utf-8") as f:
            json.dump(fold_reports, f, ensure_ascii=False, indent=2)

        with open(os.path.join(setting_dir, "fold_meta.json"), "w", encoding="utf-8") as f:
            json.dump(fold_meta, f, ensure_ascii=False, indent=2)

        with open(os.path.join(setting_dir, "feature_columns.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(feat_cols))

        df_out = df[[SUBJECT_COL] + ([TASK_COL] if TASK_COL in df.columns else []) + LABEL_COLS].copy()
        for j, ycol in enumerate(LABEL_COLS):
            df_out[f"{ycol}_prob"] = oof_prob[:, j]
            df_out[f"{ycol}_pred"] = oof_pred[:, j]
        df_out.to_csv(os.path.join(setting_dir, "oof_predictions.csv"), index=False)

        summary_rows.append({
            "setting": setting_name,
            "n_features": len(feat_cols),
            "micro_f1": overall["micro_f1"],
            "macro_f1": overall["macro_f1"],
            "minutes": set_sec / 60.0,
        })

    summary = pd.DataFrame(summary_rows).sort_values(["micro_f1", "macro_f1"], ascending=False)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()