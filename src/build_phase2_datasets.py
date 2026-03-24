import os
import json
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# CONFIG

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # final validation size out of full dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PHASE2_DIR = os.path.join(OUTPUTS_DIR, "phase2")

LEADS_PATH = os.path.join(OUTPUTS_DIR, "leads_final.csv")
MESSAGES_PATH = os.path.join(OUTPUTS_DIR, "messages_clean.csv")
SPAMBASE_PATH = os.path.join(PROJECT_ROOT, "data", "spambasedata.csv")  # checked only for reporting

NLP_DATASET_OUT = os.path.join(PHASE2_DIR, "nlp_messages_dataset.csv")
HYBRID_DATASET_OUT = os.path.join(PHASE2_DIR, "hybrid_leads_dataset.csv")

TRAIN_NLP_OUT = os.path.join(PHASE2_DIR, "train_nlp.csv")
VAL_NLP_OUT = os.path.join(PHASE2_DIR, "val_nlp.csv")
TEST_NLP_OUT = os.path.join(PHASE2_DIR, "test_nlp.csv")

TRAIN_STRUCTURED_OUT = os.path.join(PHASE2_DIR, "train_structured.csv")
VAL_STRUCTURED_OUT = os.path.join(PHASE2_DIR, "val_structured.csv")
TEST_STRUCTURED_OUT = os.path.join(PHASE2_DIR, "test_structured.csv")

SUMMARY_OUT = os.path.join(PHASE2_DIR, "split_summary.json")


# HELPERS

def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\r", " ").replace("\n", " ").strip()


def clean_binary_label(x: object) -> str:
    """
    Standardise message labels for NLP training.
    We collapse phishing/smishing into spam because Phase 2 NLP dataset
    is for spam-vs-ham text classification.
    """
    x = str(x).strip().lower()

    if x in {"ham", "legit", "not spam"}:
        return "ham"
    if x in {"spam", "phishing", "smishing"}:
        return "spam"

    return x


def safe_numeric_fill(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def clean_object_col(series: pd.Series, fill_value: str = "unknown") -> pd.Series:
    return series.fillna(fill_value).astype(str).str.strip()


def split_train_val_test(
    df: pd.DataFrame,
    stratify_col: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train / val / test split.
    val_size is defined as fraction of full dataset.
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    val_relative_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_df[stratify_col.name]
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


def make_binary_target_from_label(series: pd.Series) -> pd.Series:
    """
    For convenience later:
    ham -> 0
    spam/phishing/smishing -> 1
    """
    cleaned = series.map(clean_binary_label)
    return cleaned.map({"ham": 0, "spam": 1}).fillna(1).astype(int)


def recommended_action_to_class(series: pd.Series) -> pd.Series:
    """
    Stable integer encoding for later modelling convenience.
    """
    mapping = {
        "send_to_sales": 0,
        "needs_review": 1,
        "manual_risk_review": 2,
        "block_or_quarantine": 3,
    }
    return series.astype(str).str.strip().map(mapping)


def check_required_columns(df: pd.DataFrame, required_cols: list, df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def label_distribution(df: pd.DataFrame, col: str) -> Dict[str, int]:
    return df[col].astype(str).value_counts(dropna=False).to_dict()


# LOADERS

def load_messages_clean() -> pd.DataFrame:
    df = pd.read_csv(MESSAGES_PATH)

    required_cols = [
        "message_id",
        "source_dataset",
        "label",
        "message_text",
        "message_length_chars",
        "token_count",
        "has_url",
        "has_email",
        "has_phone",
        "urgency_score",
        "needs_sliding_window",
        "too_long_for_single_512_pass",
    ]
    check_required_columns(df, required_cols, "messages_clean.csv")

    return df


def load_leads_final() -> pd.DataFrame:
    df = pd.read_csv(LEADS_PATH)

    required_cols = [
        "lead_id",
        "company_id",
        "message_id",
        "company_name",
        "company_number",
        "company_status",
        "company_category",
        "industry",
        "company_size",
        "country",
        "location",
        "company_age_years",
        "avg_review_score",
        "review_count",
        "message_text",
        "label",
        "source_dataset",
        "message_length_chars",
        "token_count",
        "has_url",
        "has_email",
        "has_phone",
        "urgency_score",
        "needs_sliding_window",
        "too_long_for_single_512_pass",
        "generated_contact_email",
        "domain_type",
        "spam_score_rule",
        "risk_score_rule",
        "quality_score_rule",
        "recommended_action",
    ]
    check_required_columns(df, required_cols, "leads_final.csv")

    return df


# NLP DATASET PREP

def build_nlp_dataset(messages: pd.DataFrame) -> pd.DataFrame:
    nlp = messages.copy()

    nlp["message_text"] = nlp["message_text"].map(normalize_text)
    nlp["label"] = nlp["label"].map(clean_binary_label)

    nlp = nlp[nlp["message_text"] != ""].copy()
    nlp = nlp[nlp["label"].isin(["ham", "spam"])].copy()

    # drop exact duplicate text-label pairs
    nlp = nlp.drop_duplicates(subset=["message_text", "label"]).reset_index(drop=True)

    # final columns for NLP training
    nlp = nlp.rename(columns={"message_text": "text"})
    nlp["label_id"] = nlp["label"].map({"ham": 0, "spam": 1}).astype(int)

    final_cols = [
        "message_id",
        "source_dataset",
        "text",
        "label",
        "label_id",
        "message_length_chars",
        "token_count",
        "has_url",
        "has_email",
        "has_phone",
        "urgency_score",
        "needs_sliding_window",
        "too_long_for_single_512_pass",
    ]

    return nlp[final_cols].copy()


# HYBRID DATASET PREP

def build_hybrid_dataset(leads: pd.DataFrame) -> pd.DataFrame:
    hybrid = leads.copy()

    # clean text
    hybrid["message_text"] = hybrid["message_text"].map(normalize_text)
    hybrid = hybrid[hybrid["message_text"] != ""].copy()

    # clean labels
    hybrid["label_clean"] = hybrid["label"].map(clean_binary_label)
    hybrid["label_binary"] = make_binary_target_from_label(hybrid["label"])

    # numeric cleaning
    numeric_cols = [
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
        "quality_score_rule",
    ]
    for col in numeric_cols:
        hybrid[col] = safe_numeric_fill(hybrid[col], fill_value=0.0)

    # categorical cleaning
    categorical_cols = [
        "company_name",
        "company_number",
        "company_status",
        "company_category",
        "industry",
        "company_size",
        "country",
        "location",
        "source_dataset",
        "generated_contact_email",
        "domain_type",
        "recommended_action",
    ]
    for col in categorical_cols:
        hybrid[col] = clean_object_col(hybrid[col], fill_value="unknown")

    # derived helper targets for later modelling
    hybrid["recommended_action_class"] = recommended_action_to_class(hybrid["recommended_action"])

    # optional simple business-quality helper flags
    hybrid["is_active_company"] = (hybrid["company_status"].str.lower() == "active").astype(int)
    hybrid["has_reviews"] = (hybrid["review_count"] > 0).astype(int)
    hybrid["is_corporate_domain"] = (hybrid["domain_type"].str.lower() == "corporate").astype(int)
    hybrid["good_review_score_flag"] = (hybrid["avg_review_score"] >= 3.5).astype(int)
    hybrid["company_age_3plus_flag"] = (hybrid["company_age_years"] >= 3).astype(int)

    final_cols = [
        "lead_id",
        "company_id",
        "message_id",
        "company_name",
        "company_number",
        "company_status",
        "company_category",
        "industry",
        "company_size",
        "country",
        "location",
        "company_age_years",
        "avg_review_score",
        "review_count",
        "message_text",
        "label",
        "label_clean",
        "label_binary",
        "source_dataset",
        "message_length_chars",
        "token_count",
        "has_url",
        "has_email",
        "has_phone",
        "urgency_score",
        "needs_sliding_window",
        "too_long_for_single_512_pass",
        "generated_contact_email",
        "domain_type",
        "spam_score_rule",
        "risk_score_rule",
        "quality_score_rule",
        "recommended_action",
        "recommended_action_class",
        "is_active_company",
        "has_reviews",
        "is_corporate_domain",
        "good_review_score_flag",
        "company_age_3plus_flag",
    ]

    return hybrid[final_cols].copy()


# MAIN

def main() -> None:
    os.makedirs(PHASE2_DIR, exist_ok=True)

    print("Loading Phase 1 outputs...")
    messages = load_messages_clean()
    leads = load_leads_final()

    print(f"messages_clean.csv shape: {messages.shape}")
    print(f"leads_final.csv shape: {leads.shape}")

    spambase_found = os.path.exists(SPAMBASE_PATH)
    if spambase_found:
        print(f"Found spambasedata.csv at: {SPAMBASE_PATH}")
        print("Note: spambasedata.csv is NOT used in NLP text training because it has no message_text column.")
    else:
        print("spambasedata.csv not found in project data folder. This does not block Phase 2.")

    print("\nBuilding NLP dataset...")
    nlp_df = build_nlp_dataset(messages)
    print(f"NLP dataset shape: {nlp_df.shape}")

    print("Building hybrid dataset...")
    hybrid_df = build_hybrid_dataset(leads)
    print(f"Hybrid dataset shape: {hybrid_df.shape}")

    print("\nCreating NLP splits...")
    train_nlp, val_nlp, test_nlp = split_train_val_test(
        nlp_df,
        stratify_col=nlp_df["label"]
    )

    print("Creating hybrid splits...")
    train_structured, val_structured, test_structured = split_train_val_test(
        hybrid_df,
        stratify_col=hybrid_df["recommended_action"]
    )

    print("\nSaving Phase 2 outputs...")
    nlp_df.to_csv(NLP_DATASET_OUT, index=False)
    hybrid_df.to_csv(HYBRID_DATASET_OUT, index=False)

    train_nlp.to_csv(TRAIN_NLP_OUT, index=False)
    val_nlp.to_csv(VAL_NLP_OUT, index=False)
    test_nlp.to_csv(TEST_NLP_OUT, index=False)

    train_structured.to_csv(TRAIN_STRUCTURED_OUT, index=False)
    val_structured.to_csv(VAL_STRUCTURED_OUT, index=False)
    test_structured.to_csv(TEST_STRUCTURED_OUT, index=False)

    summary = {
        "config": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
        },
        "inputs": {
            "messages_clean_path": MESSAGES_PATH,
            "leads_final_path": LEADS_PATH,
            "spambasedata_path_checked": SPAMBASE_PATH,
            "spambasedata_found": spambase_found,
            "spambasedata_used": False,
            "spambasedata_reason": "Uploaded file is numeric UCI Spambase-style data with no message_text column.",
        },
        "datasets": {
            "nlp_messages_dataset_shape": list(nlp_df.shape),
            "hybrid_leads_dataset_shape": list(hybrid_df.shape),
        },
        "splits": {
            "train_nlp_shape": list(train_nlp.shape),
            "val_nlp_shape": list(val_nlp.shape),
            "test_nlp_shape": list(test_nlp.shape),
            "train_structured_shape": list(train_structured.shape),
            "val_structured_shape": list(val_structured.shape),
            "test_structured_shape": list(test_structured.shape),
        },
        "distributions": {
            "nlp_full_label_distribution": label_distribution(nlp_df, "label"),
            "train_nlp_label_distribution": label_distribution(train_nlp, "label"),
            "val_nlp_label_distribution": label_distribution(val_nlp, "label"),
            "test_nlp_label_distribution": label_distribution(test_nlp, "label"),
            "hybrid_full_action_distribution": label_distribution(hybrid_df, "recommended_action"),
            "train_structured_action_distribution": label_distribution(train_structured, "recommended_action"),
            "val_structured_action_distribution": label_distribution(val_structured, "recommended_action"),
            "test_structured_action_distribution": label_distribution(test_structured, "recommended_action"),
        },
    }

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved files:")
    print(NLP_DATASET_OUT)
    print(HYBRID_DATASET_OUT)
    print(TRAIN_NLP_OUT)
    print(VAL_NLP_OUT)
    print(TEST_NLP_OUT)
    print(TRAIN_STRUCTURED_OUT)
    print(VAL_STRUCTURED_OUT)
    print(TEST_STRUCTURED_OUT)
    print(SUMMARY_OUT)

    print("\nDone.")


if __name__ == "__main__":
    main()