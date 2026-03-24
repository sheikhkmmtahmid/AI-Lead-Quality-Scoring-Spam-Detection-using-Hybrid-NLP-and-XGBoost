import os
import json
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBRegressor


# PATHS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PHASE2_DIR = os.path.join(OUTPUTS_DIR, "phase2")
PHASE3_DIR = os.path.join(OUTPUTS_DIR, "phase3")
XGB_DIR = os.path.join(PHASE3_DIR, "xgboost")

TRAIN_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "train_structured.csv")
VAL_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "val_structured.csv")
TEST_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "test_structured.csv")

TRAIN_NLP_FEATURES_PATH = os.path.join(PHASE3_DIR, "nlp_features", "train_nlp_features.csv")
VAL_NLP_FEATURES_PATH = os.path.join(PHASE3_DIR, "nlp_features", "val_nlp_features.csv")
TEST_NLP_FEATURES_PATH = os.path.join(PHASE3_DIR, "nlp_features", "test_nlp_features.csv")

TRAIN_EMAIL_DOMAIN_FEATURES_PATH = os.path.join(PHASE3_DIR, "email_domain_features", "train_email_domain_features.csv")
VAL_EMAIL_DOMAIN_FEATURES_PATH = os.path.join(PHASE3_DIR, "email_domain_features", "val_email_domain_features.csv")
TEST_EMAIL_DOMAIN_FEATURES_PATH = os.path.join(PHASE3_DIR, "email_domain_features", "test_email_domain_features.csv")

MODEL_PATH = os.path.join(XGB_DIR, "model", "xgboost_hybrid_quality_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(XGB_DIR, "model", "feature_columns.json")

EXPLAIN_DIR = os.path.join(XGB_DIR, "explanations")
GLOBAL_DIR = os.path.join(EXPLAIN_DIR, "global")
LOCAL_DIR = os.path.join(EXPLAIN_DIR, "local")
META_DIR = os.path.join(EXPLAIN_DIR, "metadata")

BUSINESS_SUMMARY_JSON = os.path.join(GLOBAL_DIR, "business_summary.json")
TOP_FACTOR_CSV = os.path.join(GLOBAL_DIR, "top_factor_summary.csv")
GLOBAL_IMPORTANCE_PNG = os.path.join(GLOBAL_DIR, "global_feature_importance.png")
GLOBAL_SHAP_BAR_PNG = os.path.join(GLOBAL_DIR, "global_shap_bar.png")
GLOBAL_SHAP_BEESWARM_PNG = os.path.join(GLOBAL_DIR, "global_shap_beeswarm.png")

PDP_URGENCY_PNG = os.path.join(GLOBAL_DIR, "pdp_urgency_score.png")
PDP_BERT_SPAM_PNG = os.path.join(GLOBAL_DIR, "pdp_bert_prob_spam.png")
PDP_ROBERTA_SPAM_PNG = os.path.join(GLOBAL_DIR, "pdp_roberta_prob_spam.png")
PDP_COMPANY_AGE_PNG = os.path.join(GLOBAL_DIR, "pdp_company_age_years.png")
PDP_EMAIL_SYNTAX_VALID_PNG = os.path.join(GLOBAL_DIR, "pdp_email_syntax_valid.png")
PDP_DOMAIN_TRUST_SCORE_PNG = os.path.join(GLOBAL_DIR, "pdp_domain_trust_score.png")
PDP_SMTP_MAILBOX_ACCEPTED_PNG = os.path.join(GLOBAL_DIR, "pdp_smtp_mailbox_accepted.png")

TEST_BUSINESS_EXPLANATIONS_CSV = os.path.join(LOCAL_DIR, "test_business_explanations.csv")
RUN_SUMMARY_JSON = os.path.join(META_DIR, "explanation_run_summary.json")

TARGET_COL = "quality_score_rule"
SHAP_SAMPLE_SIZE = 500


# HELPERS

def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_output_dirs() -> None:
    create_dir(EXPLAIN_DIR)
    create_dir(GLOBAL_DIR)
    create_dir(LOCAL_DIR)
    create_dir(META_DIR)


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


def safe_sample_df(df: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state).copy()


def safe_feature_value(value):
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value)


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

    if "message_text" not in df.columns:
        df["message_text"] = ""

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

def build_feature_matrix(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
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
    X = X.reindex(columns=feature_columns, fill_value=0.0)

    y = sanitize_numeric_series(working[TARGET_COL])
    return X, y


# BUSINESS MAPPING

def business_label(feature: str) -> str:
    mapping = {
        "urgency_score": "Message sounds urgent or pressuring",
        "has_url": "Message contains a link",
        "has_email": "Message includes an email address",
        "has_phone": "Message includes a phone number",
        "message_length_chars": "Message length influences trust",
        "token_count": "Message size influences quality",
        "company_age_years": "Company appears more established",
        "avg_review_score": "Company has stronger public reviews",
        "review_count": "Company has more public feedback",
        "is_active_company": "Company appears active",
        "is_corporate_domain": "Uses a business email or domain",
        "good_review_score_flag": "Company review quality looks stronger",
        "company_age_3plus_flag": "Company has been around longer",
        "bert_prob_spam": "BERT sees suspicious message patterns",
        "roberta_prob_spam": "RoBERTa sees suspicious wording patterns",
        "bert_prob_ham": "BERT sees legitimate message patterns",
        "roberta_prob_ham": "RoBERTa sees legitimate wording patterns",
        "risk_score_rule": "Rules-based risk checks raised concern",
        "spam_score_rule": "Rules-based spam checks raised concern",

        "email_syntax_valid": "Email format looks valid",
        "domain_present": "An email domain is present",
        "domain_has_mx": "Domain has mail exchange records",
        "domain_has_a": "Domain has website or DNS address records",
        "smtp_reachable": "Mail server is reachable",
        "smtp_mailbox_accepted": "SMTP check suggests the mailbox can receive mail",
        "is_free_provider": "Email uses a free email provider",
        "is_disposable_provider": "Email looks disposable or temporary",
        "domain_matches_company_name": "Email domain matches the company name",
        "domain_trust_score": "Domain trust score influences lead quality",
        "smtp_response_code": "SMTP response code influences trust",
    }

    if feature in mapping:
        return mapping[feature]

    if feature.startswith("company_status_"):
        return "Company status influences trust"
    if feature.startswith("company_category_"):
        return "Company type influences lead quality"
    if feature.startswith("industry_"):
        return "Industry background influences lead quality"
    if feature.startswith("company_size_"):
        return "Company size influences lead quality"
    if feature.startswith("country_"):
        return "Country pattern influences lead quality"
    if feature.startswith("location_"):
        return "Location pattern influences lead quality"
    if feature.startswith("domain_type_"):
        return "Domain type influences trust"
    if feature.startswith("source_dataset_"):
        return "Historical source patterns influence quality"
    if feature.startswith("resolved_domain_"):
        return "Resolved domain identity influences trust"
    if feature.startswith("smtp_response_message_"):
        return "SMTP server response message influences trust"
    if feature.startswith("smtp_exception_"):
        return "SMTP exception pattern influences trust"
    if feature.startswith("mx_hosts_"):
        return "Mail server identity influences trust"
    if feature.startswith("input_contact_email_"):
        return "Input contact email pattern influences trust"
    if feature.startswith("input_website_url_"):
        return "Input website URL pattern influences trust"
    if feature.startswith("extracted_email_from_text_"):
        return "Email extracted from message text influences trust"
    if feature.startswith("extracted_url_from_text_"):
        return "URL extracted from message text influences trust"
    if feature.startswith("resolved_email_"):
        return "Resolved email identity influences trust"
    if feature.startswith("bert_emb_"):
        return "BERT text-pattern signal"
    if feature.startswith("roberta_emb_"):
        return "RoBERTa text-pattern signal"

    return feature.replace("_", " ").strip().capitalize()


def business_direction(shap_value: float) -> str:
    return "helped increase the lead score" if shap_value >= 0 else "reduced the lead score"


def score_band(score: float) -> str:
    if score >= 80:
        return "High quality"
    if score >= 60:
        return "Promising"
    if score >= 40:
        return "Needs review"
    return "Low quality"


def recommended_action_from_score(score: float) -> str:
    if score >= 80:
        return "Send to sales quickly"
    if score >= 60:
        return "Review soon and prioritise"
    if score >= 40:
        return "Review before contacting"
    return "Hold or deprioritise"


def summarise_factor_with_value(feature: str, shap_value: float, feature_value) -> str:
    return f"{business_label(feature)} ({safe_feature_value(feature_value)}) {business_direction(shap_value)}"


# GLOBAL EXPLANATIONS

def save_global_feature_importance(model: XGBRegressor, feature_columns: List[str]) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df["business_label"] = importance_df["feature"].map(business_label)
    importance_df.to_csv(TOP_FACTOR_CSV, index=False)

    top_plot = importance_df.head(15).copy().iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(top_plot["business_label"], top_plot["importance"])
    plt.xlabel("Influence on the model")
    plt.ylabel("Factor")
    plt.title("Top factors influencing lead quality decisions")
    plt.tight_layout()
    plt.savefig(GLOBAL_IMPORTANCE_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    return importance_df


def save_global_shap_plots(model: XGBRegressor, X_train: pd.DataFrame) -> pd.DataFrame:
    shap_input = safe_sample_df(X_train, SHAP_SAMPLE_SIZE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_input)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": shap_input.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df["business_label"] = shap_df["feature"].map(business_label)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_input, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(GLOBAL_SHAP_BAR_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_input, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(GLOBAL_SHAP_BEESWARM_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    return shap_df


def save_pdp_plots(model: XGBRegressor, X_train: pd.DataFrame) -> List[str]:
    plot_features = [
        ("urgency_score", PDP_URGENCY_PNG),
        ("bert_prob_spam", PDP_BERT_SPAM_PNG),
        ("roberta_prob_spam", PDP_ROBERTA_SPAM_PNG),
        ("company_age_years", PDP_COMPANY_AGE_PNG),
        ("email_syntax_valid", PDP_EMAIL_SYNTAX_VALID_PNG),
        ("domain_trust_score", PDP_DOMAIN_TRUST_SCORE_PNG),
        ("smtp_mailbox_accepted", PDP_SMTP_MAILBOX_ACCEPTED_PNG),
    ]

    saved_paths = []

    for feature_name, out_path in plot_features:
        if feature_name not in X_train.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        PartialDependenceDisplay.from_estimator(
            model,
            X_train,
            [feature_name],
            ax=ax,
        )
        ax.set_title(f"How {business_label(feature_name)} changes the lead score")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def save_business_summary(
    importance_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
) -> None:
    top_factors = importance_df.head(5)[["feature", "business_label", "importance"]].to_dict(orient="records")
    top_shap_factors = shap_df.head(5)[["feature", "business_label", "mean_abs_shap"]].to_dict(orient="records")

    summary = {
        "headline": "Lead quality explanation summary",
        "what_the_model_is_doing": (
            "The system combines message behaviour, company trust signals, "
            "email and domain validation signals, and text understanding from "
            "two language models to estimate lead quality."
        ),
        "top_factors_overall": top_factors,
        "top_factors_by_shap": top_shap_factors,
        "typical_train_score": float(np.mean(train_pred)),
        "typical_test_score": float(np.mean(test_pred)),
        "business_readout": {
            "high_quality_leads": "Usually score 80 or above and can be sent to sales quickly.",
            "review_zone_leads": "Usually score between 40 and 79 and should be reviewed before contact.",
            "low_quality_leads": "Usually score below 40 and can be deprioritised or checked carefully.",
        },
    }
    save_json(BUSINESS_SUMMARY_JSON, summary)


# LOCAL EXPLANATIONS

def build_local_business_explanations(
    raw_test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: XGBRegressor,
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    predictions = clip_quality_scores(model.predict(X_test))

    rows = []

    for idx in range(len(X_test)):
        raw_row = raw_test_df.iloc[idx]
        lead_id = int(raw_row["lead_id"])
        message_id = int(raw_row["message_id"])
        true_score = float(y_test.iloc[idx])
        pred_score = float(predictions[idx])

        row_shap = shap_values[idx]
        feature_impacts = pd.DataFrame({
            "feature": X_test.columns,
            "shap_value": row_shap,
            "feature_value": X_test.iloc[idx].values,
        })
        feature_impacts["abs_shap"] = feature_impacts["shap_value"].abs()
        feature_impacts = feature_impacts.sort_values("abs_shap", ascending=False)

        top_reasons = feature_impacts.head(5).copy()

        readable_reasons = []
        for _, r in top_reasons.head(3).iterrows():
            readable_reasons.append(
                summarise_factor_with_value(
                    feature=str(r["feature"]),
                    shap_value=float(r["shap_value"]),
                    feature_value=r["feature_value"],
                )
            )

        score_text = score_band(pred_score)
        action_text = recommended_action_from_score(pred_score)

        row_payload = {
            "lead_id": lead_id,
            "message_id": message_id,
            "true_quality_score": round(true_score, 2),
            "pred_quality_score": round(pred_score, 2),
            "score_band": score_text,
            "recommended_action": action_text,
            "top_reason_1": readable_reasons[0] if len(readable_reasons) > 0 else "",
            "top_reason_2": readable_reasons[1] if len(readable_reasons) > 1 else "",
            "top_reason_3": readable_reasons[2] if len(readable_reasons) > 2 else "",
        }

        optional_cols = [
            "company_name",
            "message_text",
            "input_contact_email",
            "input_website_url",
            "resolved_email",
            "resolved_domain",
            "smtp_response_code",
            "smtp_response_message",
            "smtp_exception",
            "mx_hosts",
        ]
        for col in optional_cols:
            if col in raw_test_df.columns:
                row_payload[col] = str(raw_row[col])

        rows.append(row_payload)

        explanation_json = {
            "lead_id": lead_id,
            "message_id": message_id,
            "lead_summary": {
                "overall_score": round(pred_score, 2),
                "quality_band": score_text,
                "recommended_action": action_text,
            },
            "record_context": {
                "company_name": str(raw_row["company_name"]) if "company_name" in raw_test_df.columns else "",
                "message_text": str(raw_row["message_text"]) if "message_text" in raw_test_df.columns else "",
                "input_contact_email": str(raw_row["input_contact_email"]) if "input_contact_email" in raw_test_df.columns else "",
                "resolved_email": str(raw_row["resolved_email"]) if "resolved_email" in raw_test_df.columns else "",
                "resolved_domain": str(raw_row["resolved_domain"]) if "resolved_domain" in raw_test_df.columns else "",
            },
            "why_the_system_scored_it_this_way": readable_reasons,
            "top_factor_table": [
                {
                    "factor": business_label(str(r["feature"])),
                    "raw_feature": str(r["feature"]),
                    "feature_value": safe_feature_value(r["feature_value"]),
                    "effect": business_direction(float(r["shap_value"])),
                    "impact_size": round(float(abs(r["shap_value"])), 4),
                }
                for _, r in top_reasons.iterrows()
            ],
        }

        out_path = os.path.join(LOCAL_DIR, f"lead_{lead_id}_business_explanation.json")
        save_json(out_path, explanation_json)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(TEST_BUSINESS_EXPLANATIONS_CSV, index=False)
    return result_df


# MAIN

def main() -> None:
    setup_output_dirs()

    print("Loading model and data...")
    check_file_exists(MODEL_PATH)
    check_file_exists(FEATURE_COLUMNS_PATH)

    model: XGBRegressor = joblib.load(MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)["feature_columns"]

    train_structured = load_structured(TRAIN_STRUCTURED_PATH, "train_structured.csv")
    val_structured = load_structured(VAL_STRUCTURED_PATH, "val_structured.csv")
    test_structured = load_structured(TEST_STRUCTURED_PATH, "test_structured.csv")

    train_nlp = load_nlp_features(TRAIN_NLP_FEATURES_PATH, "train_nlp_features.csv")
    val_nlp = load_nlp_features(VAL_NLP_FEATURES_PATH, "val_nlp_features.csv")
    test_nlp = load_nlp_features(TEST_NLP_FEATURES_PATH, "test_nlp_features.csv")

    train_email_domain = load_email_domain_features(TRAIN_EMAIL_DOMAIN_FEATURES_PATH, "train_email_domain_features.csv")
    val_email_domain = load_email_domain_features(VAL_EMAIL_DOMAIN_FEATURES_PATH, "val_email_domain_features.csv")
    test_email_domain = load_email_domain_features(TEST_EMAIL_DOMAIN_FEATURES_PATH, "test_email_domain_features.csv")

    train_df = merge_two_frames(train_structured, train_nlp, "train", "structured_plus_nlp")
    train_df = merge_two_frames(train_df, train_email_domain, "train", "plus_email_domain")

    val_df = merge_two_frames(val_structured, val_nlp, "validation", "structured_plus_nlp")
    val_df = merge_two_frames(val_df, val_email_domain, "validation", "plus_email_domain")

    test_df = merge_two_frames(test_structured, test_nlp, "test", "structured_plus_nlp")
    test_df = merge_two_frames(test_df, test_email_domain, "test", "plus_email_domain")

    X_train, y_train = build_feature_matrix(train_df, feature_columns)
    X_val, y_val = build_feature_matrix(val_df, feature_columns)
    X_test, y_test = build_feature_matrix(test_df, feature_columns)

    print("Generating global explanations...")
    importance_df = save_global_feature_importance(model, feature_columns)
    shap_df = save_global_shap_plots(model, X_train)
    saved_pdp_paths = save_pdp_plots(model, X_train)

    train_pred = clip_quality_scores(model.predict(X_train))
    test_pred = clip_quality_scores(model.predict(X_test))
    save_business_summary(importance_df, shap_df, train_pred, test_pred)

    print("Generating local business explanations for the test set...")
    local_df = build_local_business_explanations(
        raw_test_df=test_df,
        X_test=X_test,
        y_test=y_test,
        model=model,
    )

    run_summary = {
        "model_used": MODEL_PATH,
        "feature_columns_used": len(feature_columns),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "global_outputs": {
            "business_summary_json": BUSINESS_SUMMARY_JSON,
            "top_factor_csv": TOP_FACTOR_CSV,
            "global_feature_importance_png": GLOBAL_IMPORTANCE_PNG,
            "global_shap_bar_png": GLOBAL_SHAP_BAR_PNG,
            "global_shap_beeswarm_png": GLOBAL_SHAP_BEESWARM_PNG,
            "pdp_outputs": saved_pdp_paths,
        },
        "local_outputs": {
            "test_business_explanations_csv": TEST_BUSINESS_EXPLANATIONS_CSV,
            "local_json_count": int(len(local_df)),
        },
    }

    save_json(RUN_SUMMARY_JSON, run_summary)

    print("\nSaved outputs:")
    print(BUSINESS_SUMMARY_JSON)
    print(TOP_FACTOR_CSV)
    print(GLOBAL_IMPORTANCE_PNG)
    print(GLOBAL_SHAP_BAR_PNG)
    print(GLOBAL_SHAP_BEESWARM_PNG)
    for p in saved_pdp_paths:
        print(p)
    print(TEST_BUSINESS_EXPLANATIONS_CSV)
    print(RUN_SUMMARY_JSON)
    print("\nDone.")


if __name__ == "__main__":
    main()