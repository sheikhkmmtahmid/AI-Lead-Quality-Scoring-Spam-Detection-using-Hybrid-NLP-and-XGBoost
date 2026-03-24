import os
import json
from typing import List, Tuple, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# PATHS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PHASE2_DIR = os.path.join(OUTPUTS_DIR, "phase2")
PHASE3_DIR = os.path.join(OUTPUTS_DIR, "phase3")

# Structured inputs
TRAIN_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "train_structured.csv")
VAL_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "val_structured.csv")
TEST_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "test_structured.csv")

# NLP feature inputs
TRAIN_NLP_FEATURES_PATH = os.path.join(PHASE3_DIR, "nlp_features", "train_nlp_features.csv")
VAL_NLP_FEATURES_PATH = os.path.join(PHASE3_DIR, "nlp_features", "val_nlp_features.csv")
TEST_NLP_FEATURES_PATH = os.path.join(PHASE3_DIR, "nlp_features", "test_nlp_features.csv")

# Email/domain feature inputs
TRAIN_EMAIL_DOMAIN_FEATURES_PATH = os.path.join(PHASE3_DIR, "email_domain_features", "train_email_domain_features.csv")
VAL_EMAIL_DOMAIN_FEATURES_PATH = os.path.join(PHASE3_DIR, "email_domain_features", "val_email_domain_features.csv")
TEST_EMAIL_DOMAIN_FEATURES_PATH = os.path.join(PHASE3_DIR, "email_domain_features", "test_email_domain_features.csv")

# Outputs
XGB_DIR = os.path.join(PHASE3_DIR, "xgboost")
MODEL_DIR = os.path.join(XGB_DIR, "model")
METRICS_DIR = os.path.join(XGB_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(XGB_DIR, "predictions")
SHAP_DIR = os.path.join(XGB_DIR, "shap")

MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_hybrid_quality_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
TRAIN_METRICS_PATH = os.path.join(METRICS_DIR, "train_metrics.json")
VAL_METRICS_PATH = os.path.join(METRICS_DIR, "validation_metrics.json")
TEST_METRICS_PATH = os.path.join(METRICS_DIR, "test_metrics.json")
RUN_SUMMARY_PATH = os.path.join(METRICS_DIR, "run_summary.json")
FEATURE_IMPORTANCE_CSV = os.path.join(METRICS_DIR, "feature_importance.csv")

TRAIN_PRED_PATH = os.path.join(PREDICTIONS_DIR, "train_predictions.csv")
VAL_PRED_PATH = os.path.join(PREDICTIONS_DIR, "validation_predictions.csv")
TEST_PRED_PATH = os.path.join(PREDICTIONS_DIR, "test_predictions.csv")


# CONFIG

RANDOM_STATE = 42
TARGET_COL = "quality_score_rule"

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.0,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# HELPERS

def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_output_dirs() -> None:
    create_dir(XGB_DIR)
    create_dir(MODEL_DIR)
    create_dir(METRICS_DIR)
    create_dir(PREDICTIONS_DIR)
    create_dir(SHAP_DIR)


def check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def check_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def clip_quality_scores(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 100.0)


def sanitize_numeric_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.fillna(0.0)


def sanitize_categorical_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )


def deduplicate_by_keys(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    dup_count = int(df.duplicated(subset=["lead_id", "message_id"]).sum())
    if dup_count > 0:
        print(f"[WARN] {split_name}: found {dup_count} duplicate rows on lead_id/message_id. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["lead_id", "message_id"], keep="first").copy()
    return df


def prepare_id_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    df["lead_id"] = pd.to_numeric(df["lead_id"], errors="coerce")
    df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce")

    bad_rows = df["lead_id"].isna() | df["message_id"].isna()
    bad_count = int(bad_rows.sum())
    if bad_count > 0:
        raise ValueError(f"{split_name}: found {bad_count} rows with invalid lead_id/message_id")

    df["lead_id"] = df["lead_id"].astype(int)
    df["message_id"] = df["message_id"].astype(int)
    return df


# LOADERS

def load_structured(path: str, split_name: str) -> pd.DataFrame:
    check_file_exists(path)
    df = pd.read_csv(path)

    required_cols = ["lead_id", "message_id", TARGET_COL]
    check_required_columns(df, required_cols, split_name)

    df = prepare_id_columns(df, split_name)
    df = deduplicate_by_keys(df, split_name)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    bad_target = int(df[TARGET_COL].isna().sum())
    if bad_target > 0:
        raise ValueError(f"{split_name}: found {bad_target} rows with invalid target column {TARGET_COL}")

    return df.copy()


def load_nlp_features(path: str, split_name: str) -> pd.DataFrame:
    check_file_exists(path)
    df = pd.read_csv(path)

    required_cols = ["lead_id", "message_id"]
    check_required_columns(df, required_cols, split_name)

    df = prepare_id_columns(df, split_name)
    df = deduplicate_by_keys(df, split_name)

    return df.copy()


def load_email_domain_features(path: str, split_name: str) -> pd.DataFrame:
    check_file_exists(path)
    df = pd.read_csv(path)

    required_cols = ["lead_id", "message_id"]
    check_required_columns(df, required_cols, split_name)

    df = prepare_id_columns(df, split_name)
    df = deduplicate_by_keys(df, split_name)

    return df.copy()


# MERGE

def merge_two_frames(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    split_name: str,
    stage_name: str,
) -> pd.DataFrame:
    merged = left_df.merge(
        right_df,
        on=["lead_id", "message_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(left_df):
        missing_rows = len(left_df) - len(merged)
        raise ValueError(
            f"{split_name} {stage_name}: merged row count mismatch. "
            f"left_rows={len(left_df)}, merged_rows={len(merged)}, missing_after_merge={missing_rows}"
        )

    return merged.copy()


# FEATURE BUILDING

def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    working = df.copy()

    numeric_structured = [
        "company_age_years",
        "avg_review_score",
        "review_count",
        "message_length_chars",
        "token_count",
        "has_url",
        "has_email",
        "has_phone",
        "urgency_score",
        "needs_sliding_window",
        "too_long_for_single_512_pass",
        "spam_score_rule",
        "risk_score_rule",
        "is_active_company",
        "has_reviews",
        "is_corporate_domain",
        "good_review_score_flag",
        "company_age_3plus_flag",
    ]

    categorical_structured = [
        "company_status",
        "company_category",
        "industry",
        "company_size",
        "country",
        "location",
        "domain_type",
        "source_dataset",
    ]

    nlp_scalar_features = [
        "bert_pred_label_id",
        "bert_prob_ham",
        "bert_prob_spam",
        "bert_num_chunks",
        "bert_tokenized_length",
        "bert_used_sliding_window",
        "roberta_pred_label_id",
        "roberta_prob_ham",
        "roberta_prob_spam",
        "roberta_num_chunks",
        "roberta_tokenized_length",
        "roberta_used_sliding_window",
    ]

    email_domain_numeric = [
        "email_syntax_valid",
        "domain_present",
        "domain_has_mx",
        "domain_has_a",
        "smtp_reachable",
        "smtp_mailbox_accepted",
        "is_free_provider",
        "is_disposable_provider",
        "domain_matches_company_name",
        "domain_trust_score",
        "smtp_response_code",
    ]

    email_domain_categorical = [
        "domain_type",
        "resolved_domain",
        "smtp_response_message",
        "smtp_exception",
        "mx_hosts",
        "input_contact_email",
        "input_website_url",
        "extracted_email_from_text",
        "extracted_url_from_text",
        "resolved_email",
    ]

    bert_emb_cols = sorted([c for c in working.columns if c.startswith("bert_emb_")])
    roberta_emb_cols = sorted([c for c in working.columns if c.startswith("roberta_emb_")])

    all_numeric = sorted(set(
        numeric_structured +
        nlp_scalar_features +
        email_domain_numeric +
        bert_emb_cols +
        roberta_emb_cols
    ))

    all_categorical = sorted(set(categorical_structured + email_domain_categorical))

    present_numeric = [c for c in all_numeric if c in working.columns]
    present_categorical = [c for c in all_categorical if c in working.columns]

    for col in present_numeric:
        working[col] = sanitize_numeric_series(working[col])

    for col in present_categorical:
        working[col] = sanitize_categorical_series(working[col])

    num_df = working[present_numeric].copy()

    if present_categorical:
        cat_df = pd.get_dummies(
            working[present_categorical],
            columns=present_categorical,
            prefix=present_categorical,
            dummy_na=False,
        )
    else:
        cat_df = pd.DataFrame(index=working.index)

    X = pd.concat([num_df, cat_df], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if X.shape[1] == 0:
        raise ValueError("No feature columns were built. Check your input feature files.")

    y = sanitize_numeric_series(working[TARGET_COL])

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)
        return X, y, feature_columns

    feature_columns = X.columns.tolist()
    return X, y, feature_columns


# TRAIN + EVAL

def make_prediction_frame(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    out = df[["lead_id", "message_id"]].copy()

    optional_cols = [
        "email",
        "input_contact_email",
        "resolved_email",
        "resolved_domain",
        "company_name",
        "recommended_action",
    ]
    for col in optional_cols:
        if col in df.columns:
            out[col] = df[col].astype(str)

    out["true_quality_score"] = y_true
    out["pred_quality_score"] = y_pred
    out["pred_quality_score_rounded"] = np.round(y_pred, 2)

    return out


def save_metrics(path: str, split_name: str, metrics: Dict[str, float]) -> None:
    payload = {
        "split": split_name,
        "target": TARGET_COL,
        "metrics": metrics,
    }
    save_json(path, payload)


def main() -> None:
    setup_output_dirs()

    print("Loading structured, NLP, and email/domain feature files...")

    train_structured = load_structured(TRAIN_STRUCTURED_PATH, "train_structured.csv")
    val_structured = load_structured(VAL_STRUCTURED_PATH, "val_structured.csv")
    test_structured = load_structured(TEST_STRUCTURED_PATH, "test_structured.csv")

    train_nlp = load_nlp_features(TRAIN_NLP_FEATURES_PATH, "train_nlp_features.csv")
    val_nlp = load_nlp_features(VAL_NLP_FEATURES_PATH, "val_nlp_features.csv")
    test_nlp = load_nlp_features(TEST_NLP_FEATURES_PATH, "test_nlp_features.csv")

    train_email_domain = load_email_domain_features(TRAIN_EMAIL_DOMAIN_FEATURES_PATH, "train_email_domain_features.csv")
    val_email_domain = load_email_domain_features(VAL_EMAIL_DOMAIN_FEATURES_PATH, "val_email_domain_features.csv")
    test_email_domain = load_email_domain_features(TEST_EMAIL_DOMAIN_FEATURES_PATH, "test_email_domain_features.csv")

    print("Merging train split...")
    train_df = merge_two_frames(train_structured, train_nlp, "train", "structured_plus_nlp")
    train_df = merge_two_frames(train_df, train_email_domain, "train", "plus_email_domain")

    print("Merging validation split...")
    val_df = merge_two_frames(val_structured, val_nlp, "validation", "structured_plus_nlp")
    val_df = merge_two_frames(val_df, val_email_domain, "validation", "plus_email_domain")

    print("Merging test split...")
    test_df = merge_two_frames(test_structured, test_nlp, "test", "structured_plus_nlp")
    test_df = merge_two_frames(test_df, test_email_domain, "test", "plus_email_domain")

    print(f"Train merged shape: {train_df.shape}")
    print(f"Validation merged shape: {val_df.shape}")
    print(f"Test merged shape: {test_df.shape}")

    print("Building feature matrices...")
    X_train, y_train, feature_columns = build_feature_matrix(train_df)
    X_val, y_val, _ = build_feature_matrix(val_df, feature_columns=feature_columns)
    X_test, y_test, _ = build_feature_matrix(test_df, feature_columns=feature_columns)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")

    print("Training XGBoost hybrid model...")
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True,
    )

    print("Generating predictions...")
    train_pred = clip_quality_scores(model.predict(X_train))
    val_pred = clip_quality_scores(model.predict(X_val))
    test_pred = clip_quality_scores(model.predict(X_test))

    train_metrics = compute_regression_metrics(y_train.to_numpy(), train_pred)
    val_metrics = compute_regression_metrics(y_val.to_numpy(), val_pred)
    test_metrics = compute_regression_metrics(y_test.to_numpy(), test_pred)

    print("\nTrain metrics:")
    print(train_metrics)
    print("\nValidation metrics:")
    print(val_metrics)
    print("\nTest metrics:")
    print(test_metrics)

    print("Saving model and outputs...")
    joblib.dump(model, MODEL_PATH)

    save_json(FEATURE_COLUMNS_PATH, {"feature_columns": feature_columns})

    save_metrics(TRAIN_METRICS_PATH, "train", train_metrics)
    save_metrics(VAL_METRICS_PATH, "validation", val_metrics)
    save_metrics(TEST_METRICS_PATH, "test", test_metrics)

    train_pred_df = make_prediction_frame(train_df, y_train.to_numpy(), train_pred)
    val_pred_df = make_prediction_frame(val_df, y_val.to_numpy(), val_pred)
    test_pred_df = make_prediction_frame(test_df, y_test.to_numpy(), test_pred)

    train_pred_df.to_csv(TRAIN_PRED_PATH, index=False)
    val_pred_df.to_csv(VAL_PRED_PATH, index=False)
    test_pred_df.to_csv(TEST_PRED_PATH, index=False)

    feature_importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    run_summary = {
        "target": TARGET_COL,
        "xgb_params": XGB_PARAMS,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "n_features": int(len(feature_columns)),
        "model_path": MODEL_PATH,
        "structured_input_files": {
            "train": TRAIN_STRUCTURED_PATH,
            "validation": VAL_STRUCTURED_PATH,
            "test": TEST_STRUCTURED_PATH,
        },
        "nlp_input_files": {
            "train": TRAIN_NLP_FEATURES_PATH,
            "validation": VAL_NLP_FEATURES_PATH,
            "test": TEST_NLP_FEATURES_PATH,
        },
        "email_domain_input_files": {
            "train": TRAIN_EMAIL_DOMAIN_FEATURES_PATH,
            "validation": VAL_EMAIL_DOMAIN_FEATURES_PATH,
            "test": TEST_EMAIL_DOMAIN_FEATURES_PATH,
        },
        "prediction_files": {
            "train": TRAIN_PRED_PATH,
            "validation": VAL_PRED_PATH,
            "test": TEST_PRED_PATH,
        },
        "metrics_files": {
            "train": TRAIN_METRICS_PATH,
            "validation": VAL_METRICS_PATH,
            "test": TEST_METRICS_PATH,
        },
        "feature_importance_csv": FEATURE_IMPORTANCE_CSV,
    }
    save_json(RUN_SUMMARY_PATH, run_summary)

    print("\nSaved files:")
    print(MODEL_PATH)
    print(FEATURE_COLUMNS_PATH)
    print(TRAIN_METRICS_PATH)
    print(VAL_METRICS_PATH)
    print(TEST_METRICS_PATH)
    print(TRAIN_PRED_PATH)
    print(VAL_PRED_PATH)
    print(TEST_PRED_PATH)
    print(FEATURE_IMPORTANCE_CSV)
    print(RUN_SUMMARY_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()