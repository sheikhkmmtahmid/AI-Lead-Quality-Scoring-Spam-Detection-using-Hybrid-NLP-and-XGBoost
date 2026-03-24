"""
src/retrain_from_feedback.py

Retrains BERT, RoBERTa, and XGBoost using accumulated human feedback stored in
outputs/feedback.db.

Pipeline:
  1. Load feedback rows (joined with their original scored_lead inputs)
  2. Fine-tune BERT on (message_text, actual_spam_label) pairs
  3. Fine-tune RoBERTa on (message_text, actual_spam_label) pairs
  4. Run the updated models on feedback rows to produce fresh embeddings/probs
  5. Load original XGBoost training data (phase2 + phase3 artefacts)
  6. Build a feature row for each feedback entry using the same schema
  7. Combine original + feedback rows and retrain XGBoost from scratch
  8. Save all updated models in-place (API picks them up on next restart)

Usage:
    # from project root
    python src/retrain_from_feedback.py
    python src/retrain_from_feedback.py --min-nlp 30 --min-xgb 10
    python src/retrain_from_feedback.py --dry-run
"""

import os
import re
import sys
import json
import shutil
import sqlite3
import argparse
import warnings
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ── PATHS ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH               = os.path.join(PROJECT_ROOT, "outputs", "feedback.db")
PHASE2_DIR            = os.path.join(PROJECT_ROOT, "outputs", "phase2")
PHASE3_DIR            = os.path.join(PROJECT_ROOT, "outputs", "phase3")

BERT_MODEL_DIR        = os.path.join(PHASE3_DIR, "bert",    "model")
BERT_TOKENIZER_DIR    = os.path.join(PHASE3_DIR, "bert",    "tokenizer")
ROBERTA_MODEL_DIR     = os.path.join(PHASE3_DIR, "roberta", "model")
ROBERTA_TOKENIZER_DIR = os.path.join(PHASE3_DIR, "roberta", "tokenizer")

XGB_MODEL_PATH        = os.path.join(PHASE3_DIR, "xgboost", "model", "xgboost_hybrid_quality_model.joblib")
FEATURE_COLUMNS_PATH  = os.path.join(PHASE3_DIR, "xgboost", "model", "feature_columns.json")

TRAIN_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "train_structured.csv")
TRAIN_NLP_PATH        = os.path.join(PHASE3_DIR, "nlp_features",        "train_nlp_features.csv")
TRAIN_EMAIL_PATH      = os.path.join(PHASE3_DIR, "email_domain_features", "train_email_domain_features.csv")


# ── CONFIG ────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_MAX_LENGTH       = 256
SLIDING_WINDOW_STRIDE = 192

NLP_FINETUNE_EPOCHS   = 2
NLP_LEARNING_RATE     = 2e-5
NLP_BATCH_SIZE        = 8
NLP_MAX_GRAD_NORM     = 1.0

XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.03,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       2.0,
    "random_state":     42,
    "tree_method":      "hist",
    "n_jobs":           -1,
}

# Map human feedback labels to a 0-1 quality score for XGBoost training
QUALITY_LABEL_TO_SCORE = {
    "good_lead": 0.85,
    "bad_lead":  0.20,
    "spam":      0.05,
}

# How many old model backups to keep per model
MAX_BACKUPS = 3


# ── MODEL BACKUP ─────────────────────────────────────────────────────────────

def backup_model(model_path: str, max_backups: int = MAX_BACKUPS):
    """
    Copy model_path into a timestamped backup next to the original.
    Keeps only the most recent `max_backups` copies.

    Works for both files (e.g. .joblib) and directories (e.g. safetensors folder).
    """
    if not os.path.exists(model_path):
        return  # nothing to back up

    ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parent     = os.path.dirname(model_path)
    base_name  = os.path.basename(model_path)
    backup_dir = os.path.join(parent, "backups")
    os.makedirs(backup_dir, exist_ok=True)

    # Build backup destination path
    name, ext  = os.path.splitext(base_name)
    dest       = os.path.join(backup_dir, f"{name}_backup_{ts}{ext}")

    if os.path.isdir(model_path):
        shutil.copytree(model_path, dest)
    else:
        shutil.copy2(model_path, dest)

    print(f"  [Backup] {base_name} → backups/{os.path.basename(dest)}")

    # Prune old backups — keep only the most recent `max_backups`
    pattern   = f"{name}_backup_"
    all_backups = sorted([
        os.path.join(backup_dir, f)
        for f in os.listdir(backup_dir)
        if f.startswith(pattern)
    ])
    for old in all_backups[:-max_backups]:
        if os.path.isdir(old):
            shutil.rmtree(old)
        else:
            os.remove(old)
        print(f"  [Backup] Removed old backup: {os.path.basename(old)}")


# ── DATABASE ──────────────────────────────────────────────────────────────────

def load_feedback_from_db() -> pd.DataFrame:
    """Load all feedback rows joined with their original scored_lead inputs."""
    if not os.path.exists(DB_PATH):
        print(f"[ERROR] feedback.db not found at {DB_PATH}")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            f.feedback_id,
            f.actual_quality_label,
            f.actual_spam_label,
            f.notes,
            s.lead_log_id,
            s.message_text,
            s.company_name,
            s.contact_email,
            s.company_age_years,
            s.avg_review_score,
            s.review_count,
            s.company_status,
            s.company_category,
            s.industry,
            s.company_size,
            s.country,
            s.domain_type,
            s.spam_score_rule,
            s.risk_score_rule,
            s.domain_trust_score,
            s.smtp_mailbox_accepted,
            s.is_disposable_provider
        FROM feedback f
        JOIN scored_leads s ON f.lead_log_id = s.lead_log_id
        WHERE f.actual_quality_label IS NOT NULL
          AND s.message_text IS NOT NULL
          AND TRIM(s.message_text) != ''
    """, con)
    con.close()
    return df


# ── MESSAGE FEATURE EXTRACTION ────────────────────────────────────────────────

_URL_PAT   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_PAT = re.compile(r"\b[A-Z0-9._%+\-']+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PAT = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")
_URGENT    = ["urgent","immediately","act now","limited offer","limited time","asap","now",
              "click here","verify","confirm","final notice","response needed",
              "exclusive offer","won","winner","claim"]
_SPAM_T    = ["free","offer","winner","guaranteed","click","act now","claim","prize",
              "limited","urgent","verify","confirm","bonus"]


def extract_message_features(text: str) -> dict:
    text  = text.strip()
    lower = text.lower()
    toks  = max(1, len(text.split()))
    return {
        "message_length_chars":          len(text),
        "token_count":                   toks,
        "has_url":                       int(bool(_URL_PAT.search(text))),
        "has_email":                     int(bool(_EMAIL_PAT.search(text))),
        "has_phone":                     int(bool(_PHONE_PAT.search(text))),
        "urgency_score":                 min(1.0, sum(t in lower for t in _URGENT) / 4.0),
        "needs_sliding_window":          int(toks > BASE_MAX_LENGTH),
        "too_long_for_single_512_pass":  int(toks > 512),
        "spam_hits":                     sum(t in lower for t in _SPAM_T),
    }


# ── NLP INFERENCE (sliding window, identical to api/main.py) ─────────────────

def _build_chunk_inputs(text: str, tokenizer) -> dict:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tok_len   = len(token_ids)

    if tok_len <= BASE_MAX_LENGTH:
        enc = tokenizer(text, truncation=True, padding="max_length",
                        max_length=BASE_MAX_LENGTH, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "num_chunks": 1, "tokenized_length": tok_len, "used_sliding_window": 0}

    special      = tokenizer.num_special_tokens_to_add(pair=False)
    content_max  = BASE_MAX_LENGTH - special
    cls_id, sep_id, pad_id = (tokenizer.cls_token_id,
                               tokenizer.sep_token_id,
                               tokenizer.pad_token_id)

    chunk_ids, chunk_masks = [], []
    start = 0
    while start < tok_len:
        end   = min(start + content_max, tok_len)
        chunk = token_ids[start:end]
        full  = [cls_id] + chunk + [sep_id]
        mask  = [1] * len(full)
        pad   = BASE_MAX_LENGTH - len(full)
        if pad < 0:
            full, mask = full[:BASE_MAX_LENGTH], mask[:BASE_MAX_LENGTH]
        else:
            full += [pad_id] * pad
            mask += [0]      * pad
        chunk_ids.append(full)
        chunk_masks.append(mask)
        if end >= tok_len:
            break
        start += SLIDING_WINDOW_STRIDE

    return {"input_ids":      torch.tensor(chunk_ids,   dtype=torch.long),
            "attention_mask": torch.tensor(chunk_masks, dtype=torch.long),
            "num_chunks": len(chunk_ids), "tokenized_length": tok_len, "used_sliding_window": 1}


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def run_text_model(text: str, tokenizer, model) -> dict:
    chunked = _build_chunk_inputs(text, tokenizer)
    ids     = chunked["input_ids"].to(DEVICE)
    masks   = chunked["attention_mask"].to(DEVICE)
    out     = model(input_ids=ids, attention_mask=masks,
                    output_hidden_states=True, return_dict=True)
    probs   = F.softmax(out.logits, dim=1)
    pooled  = _mean_pool(out.hidden_states[-1], masks)
    agg_p   = torch.clamp(probs.mean(0, keepdim=True), 1e-6, 1.0)
    agg_e   = pooled.mean(0, keepdim=True)
    return {
        "pred_label_id":       int(torch.argmax(agg_p, dim=1).cpu().item()),
        "prob_ham":            float(agg_p[0, 0].cpu()),
        "prob_spam":           float(agg_p[0, 1].cpu()),
        "num_chunks":          int(chunked["num_chunks"]),
        "tokenized_length":    int(chunked["tokenized_length"]),
        "used_sliding_window": int(chunked["used_sliding_window"]),
        "embedding":           agg_e.cpu().numpy()[0],
    }


# ── PYTORCH DATASET FOR FINE-TUNING ──────────────────────────────────────────

class SpamFeedbackDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=BASE_MAX_LENGTH, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── FINE-TUNE BERT / ROBERTA ──────────────────────────────────────────────────

def fine_tune_nlp_model(
    model_dir: str,
    tokenizer_dir: str,
    texts: List[str],
    labels: List[int],
    model_name: str = "model",
):
    """Fine-tune a classification model on feedback spam/ham labels and save in-place."""
    print(f"\n[{model_name}] Fine-tuning on {len(texts)} feedback samples  (device={DEVICE})")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_hidden_states=True
    ).to(DEVICE)
    model.train()

    dataset    = SpamFeedbackDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=NLP_BATCH_SIZE, shuffle=True)

    total_steps  = len(dataloader) * NLP_FINETUNE_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))

    optimizer = AdamW(model.parameters(), lr=NLP_LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    for epoch in range(1, NLP_FINETUNE_EPOCHS + 1):
        total_loss, correct = 0.0, 0
        for batch in dataloader:
            optimizer.zero_grad()
            ids     = batch["input_ids"].to(DEVICE)
            masks   = batch["attention_mask"].to(DEVICE)
            blabels = batch["labels"].to(DEVICE)
            out     = model(input_ids=ids, attention_mask=masks,
                            labels=blabels, return_dict=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), NLP_MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            correct    += (out.logits.argmax(dim=1) == blabels).sum().item()

        acc = correct / len(dataset)
        print(f"  Epoch {epoch}/{NLP_FINETUNE_EPOCHS}  "
              f"loss={total_loss / len(dataloader):.4f}  acc={acc:.3f}")

    backup_model(os.path.join(model_dir, "model.safetensors"))
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"  [{model_name}] Saved updated weights → {model_dir}")
    return model, tokenizer


# ── BUILD FEEDBACK FEATURE ROWS FOR XGBOOST ──────────────────────────────────

def build_feedback_xgb_rows(
    feedback_df: pd.DataFrame,
    feature_columns: List[str],
    bert_tokenizer,
    bert_model,
    roberta_tokenizer,
    roberta_model,
) -> pd.DataFrame:
    """
    Reconstruct XGBoost feature vectors for feedback rows.
    Uses the same schema as training (reindex fills missing columns with 0).
    """
    rows = []
    total = len(feedback_df)

    for i, (_, row) in enumerate(feedback_df.iterrows(), 1):
        if i % 10 == 0 or i == total:
            print(f"  Building feedback features: {i}/{total}", end="\r")

        text = str(row["message_text"])
        mf   = extract_message_features(text)

        bert_r    = run_text_model(text, bert_tokenizer,    bert_model)
        roberta_r = run_text_model(text, roberta_tokenizer, roberta_model)

        # Stored structured fields
        status     = str(row.get("company_status",   "unknown") or "unknown").lower()
        dom_type   = str(row.get("domain_type",      "unknown") or "unknown").lower()
        age        = float(row.get("company_age_years",  0) or 0)
        avg_score  = float(row.get("avg_review_score",   0) or 0)
        rev_cnt    = int(row.get("review_count",         0) or 0)

        feat = {col: 0.0 for col in feature_columns}

        # Numeric features
        numeric = {
            "company_age_years":            age,
            "avg_review_score":             avg_score,
            "review_count":                 rev_cnt,
            "message_length_chars":         mf["message_length_chars"],
            "token_count":                  mf["token_count"],
            "has_url":                      mf["has_url"],
            "has_email":                    mf["has_email"],
            "has_phone":                    mf["has_phone"],
            "urgency_score":                mf["urgency_score"],
            "needs_sliding_window":         mf["needs_sliding_window"],
            "too_long_for_single_512_pass": mf["too_long_for_single_512_pass"],
            "spam_score_rule":              float(row.get("spam_score_rule",  0) or 0),
            "risk_score_rule":              float(row.get("risk_score_rule",  0) or 0),
            # Derived flags
            "is_active_company":            int(status == "active"),
            "has_reviews":                  int(rev_cnt > 0),
            "is_corporate_domain":          int(dom_type == "corporate"),
            "good_review_score_flag":       int(avg_score >= 3.5),
            "company_age_3plus_flag":       int(age >= 3),
            # NLP scalars from updated models
            "bert_pred_label_id":           bert_r["pred_label_id"],
            "bert_prob_ham":                bert_r["prob_ham"],
            "bert_prob_spam":               bert_r["prob_spam"],
            "bert_num_chunks":              bert_r["num_chunks"],
            "bert_tokenized_length":        bert_r["tokenized_length"],
            "bert_used_sliding_window":     bert_r["used_sliding_window"],
            "roberta_pred_label_id":        roberta_r["pred_label_id"],
            "roberta_prob_ham":             roberta_r["prob_ham"],
            "roberta_prob_spam":            roberta_r["prob_spam"],
            "roberta_num_chunks":           roberta_r["num_chunks"],
            "roberta_tokenized_length":     roberta_r["tokenized_length"],
            "roberta_used_sliding_window":  roberta_r["used_sliding_window"],
            # Email/domain: stored subset; remainder stays 0
            "domain_trust_score":           float(row.get("domain_trust_score",   0) or 0),
            "smtp_mailbox_accepted":        int(row.get("smtp_mailbox_accepted",   0) or 0),
            "is_disposable_provider":       int(row.get("is_disposable_provider", 0) or 0),
        }

        for k, v in numeric.items():
            if k in feat:
                feat[k] = v

        # BERT embeddings (768-dim)
        for idx, v in enumerate(bert_r["embedding"]):
            key = f"bert_emb_{idx}"
            if key in feat:
                feat[key] = float(v)

        # RoBERTa embeddings (768-dim)
        for idx, v in enumerate(roberta_r["embedding"]):
            key = f"roberta_emb_{idx}"
            if key in feat:
                feat[key] = float(v)

        # Categorical one-hot flags
        for cat_key in [
            f"company_status_{status}",
            f"company_category_{str(row.get('company_category', 'unknown') or 'unknown')}",
            f"industry_{str(row.get('industry', 'unknown') or 'unknown')}",
            f"company_size_{str(row.get('company_size', 'unknown') or 'unknown')}",
            f"country_{str(row.get('country', 'unknown') or 'unknown')}",
            f"domain_type_{dom_type}",
        ]:
            if cat_key in feat:
                feat[cat_key] = 1

        rows.append(feat)

    print()
    return pd.DataFrame(rows, columns=feature_columns)


# ── LOAD ORIGINAL TRAINING DATA ───────────────────────────────────────────────

def load_original_train_data(feature_columns: List[str]):
    """
    Load and merge the original phase2/3 training splits into the XGBoost feature schema.
    Uses the same build_feature_matrix logic as train_xgboost_hybrid.py.
    """
    for path in [TRAIN_STRUCTURED_PATH, TRAIN_NLP_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required training file not found: {path}")

    structured = pd.read_csv(TRAIN_STRUCTURED_PATH)
    nlp        = pd.read_csv(TRAIN_NLP_PATH)

    df = structured.merge(nlp, on=["lead_id", "message_id"], how="inner")

    # Email domain features are optional (may not exist in all setups)
    if os.path.exists(TRAIN_EMAIL_PATH):
        email = pd.read_csv(TRAIN_EMAIL_PATH)
        df    = df.merge(email, on=["lead_id", "message_id"], how="inner", suffixes=("", "_email"))
    else:
        print(f"  [WARN] Email domain features not found at {TRAIN_EMAIL_PATH} — skipping.")

    # Derived flags (same as train_xgboost_hybrid.py)
    df["is_active_company"]     = (df["company_status"].str.lower() == "active").astype(int)
    df["has_reviews"]           = (df["review_count"] > 0).astype(int)
    df["is_corporate_domain"]   = (df["domain_type"].str.lower() == "corporate").astype(int)
    df["good_review_score_flag"] = (df["avg_review_score"] >= 3.5).astype(int)
    df["company_age_3plus_flag"] = (df["company_age_years"] >= 3).astype(int)

    # Categorical one-hot encoding
    cat_cols = [
        "company_status", "company_category", "industry", "company_size",
        "country", "location", "domain_type", "source_dataset",
        "resolved_domain", "smtp_response_message", "smtp_exception",
        "mx_hosts", "input_contact_email", "input_website_url",
        "extracted_email_from_text", "extracted_url_from_text", "resolved_email",
    ]
    present_cats = [c for c in cat_cols if c in df.columns]
    if present_cats:
        dummies = pd.get_dummies(df[present_cats], columns=present_cats,
                                 prefix=present_cats, dummy_na=False)
        df = pd.concat([df, dummies], axis=1)

    X = df.reindex(columns=feature_columns, fill_value=0.0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(df["quality_score_rule"], errors="coerce").fillna(0.5)

    return X, y


# ── RETRAIN XGBOOST ───────────────────────────────────────────────────────────

def retrain_xgboost(
    X_orig: pd.DataFrame,
    y_orig: pd.Series,
    X_feedback: pd.DataFrame,
    y_feedback: pd.Series,
):
    X_all = pd.concat([X_orig, X_feedback], ignore_index=True)
    y_all = pd.concat([y_orig, y_feedback], ignore_index=True)

    print(f"\n[XGBoost] Retraining on {len(X_orig)} original + "
          f"{len(X_feedback)} feedback rows = {len(X_all)} total")

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_all, y_all, verbose=False)

    # Sanity check on feedback rows only
    preds_fb = model.predict(X_feedback)
    mae_fb   = float(np.abs(preds_fb - y_feedback.values).mean())
    print(f"  Feedback MAE (sanity check): {mae_fb:.4f}")

    # Sanity check on original rows
    preds_orig = model.predict(X_orig)
    mae_orig   = float(np.abs(preds_orig - y_orig.values).mean())
    print(f"  Original train MAE:          {mae_orig:.4f}")

    backup_model(XGB_MODEL_PATH)
    joblib.dump(model, XGB_MODEL_PATH)
    print(f"  [XGBoost] Saved updated model → {XGB_MODEL_PATH}")
    return model


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retrain BERT, RoBERTa, and XGBoost from human feedback."
    )
    parser.add_argument("--min-nlp", type=int, default=50,
                        help="Min feedback rows with spam/ham label to fine-tune NLP models (default: 50)")
    parser.add_argument("--min-xgb", type=int, default=20,
                        help="Min feedback rows to retrain XGBoost (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only — do not retrain anything")
    args = parser.parse_args()

    print("=" * 65)
    print("  FEEDBACK-DRIVEN RETRAINING")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # ── 1. Load feedback
    feedback_df = load_feedback_from_db()
    print(f"\nFeedback rows loaded: {len(feedback_df)}")

    if feedback_df.empty:
        print("[INFO] No feedback found in the database. Nothing to retrain.")
        return

    quality_counts = feedback_df["actual_quality_label"].value_counts().to_dict()
    spam_counts    = feedback_df["actual_spam_label"].value_counts().to_dict()
    print(f"  Quality labels : {quality_counts}")
    print(f"  Spam labels    : {spam_counts}")

    # NLP rows = those with a valid spam/ham label
    nlp_df = feedback_df[
        feedback_df["actual_spam_label"].isin(["spam", "ham"])
    ].copy()
    print(f"  NLP-eligible rows (have spam/ham label): {len(nlp_df)}")

    # XGBoost rows = those with a valid quality label
    xgb_df = feedback_df[
        feedback_df["actual_quality_label"].isin(QUALITY_LABEL_TO_SCORE.keys())
    ].copy()
    xgb_df["derived_quality_score"] = xgb_df["actual_quality_label"].map(QUALITY_LABEL_TO_SCORE)
    print(f"  XGBoost-eligible rows (have quality label): {len(xgb_df)}")

    if args.dry_run:
        print("\n[DRY RUN] No models were retrained.")
        return

    # ── Check required model artefacts exist
    for path in [FEATURE_COLUMNS_PATH, XGB_MODEL_PATH,
                 BERT_MODEL_DIR, BERT_TOKENIZER_DIR,
                 ROBERTA_MODEL_DIR, ROBERTA_TOKENIZER_DIR]:
        if not os.path.exists(path):
            print(f"[ERROR] Required artefact not found: {path}")
            sys.exit(1)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_columns = json.load(f)["feature_columns"]

    bert_model = bert_tokenizer = roberta_model = roberta_tokenizer = None

    # ── 2. Fine-tune BERT
    if len(nlp_df) >= args.min_nlp:
        spam_labels = nlp_df["actual_spam_label"].map({"spam": 1, "ham": 0}).tolist()
        bert_model, bert_tokenizer = fine_tune_nlp_model(
            BERT_MODEL_DIR, BERT_TOKENIZER_DIR,
            nlp_df["message_text"].tolist(), spam_labels,
            model_name="BERT",
        )
    else:
        print(f"\n[BERT]    Skipped — need {args.min_nlp} NLP rows, have {len(nlp_df)}.")

    # ── 3. Fine-tune RoBERTa
    if len(nlp_df) >= args.min_nlp:
        spam_labels = nlp_df["actual_spam_label"].map({"spam": 1, "ham": 0}).tolist()
        roberta_model, roberta_tokenizer = fine_tune_nlp_model(
            ROBERTA_MODEL_DIR, ROBERTA_TOKENIZER_DIR,
            nlp_df["message_text"].tolist(), spam_labels,
            model_name="RoBERTa",
        )

    # ── 4–7. Retrain XGBoost
    if len(xgb_df) >= args.min_xgb:

        # Load NLP models for embedding generation (fine-tuned if available, else from disk)
        if bert_model is None:
            print("\n[XGBoost] Loading BERT for embedding generation...")
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_DIR, use_fast=True)
            bert_model = AutoModelForSequenceClassification.from_pretrained(
                BERT_MODEL_DIR, output_hidden_states=True
            ).to(DEVICE)

        if roberta_model is None:
            print("[XGBoost] Loading RoBERTa for embedding generation...")
            roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_TOKENIZER_DIR, use_fast=True)
            roberta_model = AutoModelForSequenceClassification.from_pretrained(
                ROBERTA_MODEL_DIR, output_hidden_states=True
            ).to(DEVICE)

        bert_model.eval()
        roberta_model.eval()

        print(f"\n[XGBoost] Building feature vectors for {len(xgb_df)} feedback rows...")
        X_feedback = build_feedback_xgb_rows(
            xgb_df, feature_columns,
            bert_tokenizer, bert_model,
            roberta_tokenizer, roberta_model,
        )
        y_feedback = xgb_df["derived_quality_score"].reset_index(drop=True)

        print("[XGBoost] Loading original training data...")
        X_orig, y_orig = load_original_train_data(feature_columns)

        retrain_xgboost(X_orig, y_orig, X_feedback, y_feedback)

    else:
        print(f"\n[XGBoost] Skipped — need {args.min_xgb} XGBoost-eligible rows, "
              f"have {len(xgb_df)}.")

    print("\n" + "=" * 65)
    print("  Retraining complete.")
    print("  Restart the API server to load the updated models.")
    print("=" * 65)


if __name__ == "__main__":
    main()
