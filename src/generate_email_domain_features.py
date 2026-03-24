import os
import json
from typing import List, Dict, Tuple

import pandas as pd

from email_domain_features import generate_email_domain_feature_frame


# PATHS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PHASE2_DIR = os.path.join(OUTPUTS_DIR, "phase2")
PHASE3_DIR = os.path.join(OUTPUTS_DIR, "phase3")

TRAIN_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "train_structured.csv")
VAL_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "val_structured.csv")
TEST_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "test_structured.csv")

EMAIL_DOMAIN_DIR = os.path.join(PHASE3_DIR, "email_domain_features")
TRAIN_OUT = os.path.join(EMAIL_DOMAIN_DIR, "train_email_domain_features.csv")
VAL_OUT = os.path.join(EMAIL_DOMAIN_DIR, "val_email_domain_features.csv")
TEST_OUT = os.path.join(EMAIL_DOMAIN_DIR, "test_email_domain_features.csv")
SUMMARY_OUT = os.path.join(EMAIL_DOMAIN_DIR, "email_domain_features_summary.json")


# CONFIG

ENABLE_SMTP_CHECK = True

# These are the primary expected columns in your structured files
COMPANY_NAME_COL = "company_name"
MESSAGE_TEXT_COL = "message_text"
LEAD_ID_COL = "lead_id"
MESSAGE_ID_COL = "message_id"


# HELPERS
def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def load_structured_split(path: str, split_name: str) -> pd.DataFrame:
    check_file_exists(path)
    df = pd.read_csv(path)

    required_cols = [COMPANY_NAME_COL, MESSAGE_TEXT_COL]
    check_required_columns(df, required_cols, split_name)

    if LEAD_ID_COL in df.columns:
        df[LEAD_ID_COL] = pd.to_numeric(df[LEAD_ID_COL], errors="coerce")
    if MESSAGE_ID_COL in df.columns:
        df[MESSAGE_ID_COL] = pd.to_numeric(df[MESSAGE_ID_COL], errors="coerce")

    df[COMPANY_NAME_COL] = df[COMPANY_NAME_COL].fillna("").astype(str).str.strip()
    df[MESSAGE_TEXT_COL] = df[MESSAGE_TEXT_COL].fillna("").astype(str).str.strip()

    return df.copy()


def summarise_feature_frame(df: pd.DataFrame) -> Dict:
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    for col in [
        "email_syntax_valid",
        "domain_present",
        "domain_has_mx",
        "domain_has_a",
        "smtp_reachable",
        "smtp_mailbox_accepted",
        "is_free_provider",
        "is_disposable_provider",
        "domain_matches_company_name",
        "domain_type",
    ]:
        if col in df.columns:
            if df[col].dtype == "object":
                summary[f"{col}_value_counts"] = df[col].fillna("null").value_counts().to_dict()
            else:
                summary[f"{col}_value_counts"] = (
                    df[col].fillna(-1).astype(str).value_counts().to_dict()
                )

    if "domain_trust_score" in df.columns:
        summary["domain_trust_score_stats"] = {
            "min": float(df["domain_trust_score"].min()),
            "max": float(df["domain_trust_score"].max()),
            "mean": float(df["domain_trust_score"].mean()),
        }

    return summary


# MAIN

def main() -> None:
    create_dir(EMAIL_DOMAIN_DIR)

    print("Loading structured splits...")
    train_df = load_structured_split(TRAIN_STRUCTURED_PATH, "train_structured.csv")
    val_df = load_structured_split(VAL_STRUCTURED_PATH, "val_structured.csv")
    test_df = load_structured_split(TEST_STRUCTURED_PATH, "test_structured.csv")

    print(f"Train structured shape: {train_df.shape}")
    print(f"Validation structured shape: {val_df.shape}")
    print(f"Test structured shape: {test_df.shape}")

    print("\nGenerating email/domain features for train...")
    train_features = generate_email_domain_feature_frame(
        df=train_df,
        company_name_col=COMPANY_NAME_COL,
        message_text_col=MESSAGE_TEXT_COL,
        lead_id_col=LEAD_ID_COL if LEAD_ID_COL in train_df.columns else None,
        message_id_col=MESSAGE_ID_COL if MESSAGE_ID_COL in train_df.columns else None,
        enable_smtp_check=ENABLE_SMTP_CHECK,
    )

    print("Generating email/domain features for validation...")
    val_features = generate_email_domain_feature_frame(
        df=val_df,
        company_name_col=COMPANY_NAME_COL,
        message_text_col=MESSAGE_TEXT_COL,
        lead_id_col=LEAD_ID_COL if LEAD_ID_COL in val_df.columns else None,
        message_id_col=MESSAGE_ID_COL if MESSAGE_ID_COL in val_df.columns else None,
        enable_smtp_check=ENABLE_SMTP_CHECK,
    )

    print("Generating email/domain features for test...")
    test_features = generate_email_domain_feature_frame(
        df=test_df,
        company_name_col=COMPANY_NAME_COL,
        message_text_col=MESSAGE_TEXT_COL,
        lead_id_col=LEAD_ID_COL if LEAD_ID_COL in test_df.columns else None,
        message_id_col=MESSAGE_ID_COL if MESSAGE_ID_COL in test_df.columns else None,
        enable_smtp_check=ENABLE_SMTP_CHECK,
    )

    print("\nSaving outputs...")
    train_features.to_csv(TRAIN_OUT, index=False)
    val_features.to_csv(VAL_OUT, index=False)
    test_features.to_csv(TEST_OUT, index=False)

    summary = {
        "config": {
            "enable_smtp_check": ENABLE_SMTP_CHECK,
            "company_name_col": COMPANY_NAME_COL,
            "message_text_col": MESSAGE_TEXT_COL,
            "lead_id_col": LEAD_ID_COL,
            "message_id_col": MESSAGE_ID_COL,
        },
        "files_created": {
            "train": TRAIN_OUT,
            "validation": VAL_OUT,
            "test": TEST_OUT,
        },
        "train_summary": summarise_feature_frame(train_features),
        "validation_summary": summarise_feature_frame(val_features),
        "test_summary": summarise_feature_frame(test_features),
    }

    save_json(SUMMARY_OUT, summary)

    print("Done.")
    print(TRAIN_OUT)
    print(VAL_OUT)
    print(TEST_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()