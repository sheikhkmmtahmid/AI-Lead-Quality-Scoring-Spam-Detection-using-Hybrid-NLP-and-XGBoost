import os
import re
import json
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.email_domain_features import build_email_domain_features


# PATHS

#PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
INDEX_HTML_PATH = os.path.join(STATIC_DIR, "index.html")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PHASE3_DIR = os.path.join(OUTPUTS_DIR, "phase3")

XGB_MODEL_PATH = os.path.join(PHASE3_DIR, "xgboost", "model", "xgboost_hybrid_quality_model.joblib")
FEATURE_COLUMNS_PATH = os.path.join(PHASE3_DIR, "xgboost", "model", "feature_columns.json")

BERT_MODEL_DIR = os.path.join(PHASE3_DIR, "bert", "model")
BERT_TOKENIZER_DIR = os.path.join(PHASE3_DIR, "bert", "tokenizer")

ROBERTA_MODEL_DIR = os.path.join(PHASE3_DIR, "roberta", "model")
ROBERTA_TOKENIZER_DIR = os.path.join(PHASE3_DIR, "roberta", "tokenizer")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DB_PATH              = os.path.join(PROJECT_ROOT, "outputs", "feedback.db")
RETRAIN_STATE_PATH   = os.path.join(PROJECT_ROOT, "outputs", "retrain_state.json")
RETRAIN_SCRIPT_PATH  = os.path.join(PROJECT_ROOT, "src", "retrain_from_feedback.py")

# Trigger retraining automatically after this many NEW feedback rows since last retrain
AUTO_RETRAIN_THRESHOLD = 100

# Scheduled nightly retraining — set to False to disable
SCHEDULED_RETRAIN_ENABLED = True
SCHEDULED_RETRAIN_TIME    = "02:00"   # 24-hour HH:MM (server local time)
SCHEDULED_RETRAIN_MIN_XGB = 20        # minimum new feedback rows required to actually retrain


# CONFIG

BASE_MAX_LENGTH = 256
SLIDING_WINDOW_STRIDE = 192
AUTO_OPEN_BROWSER = True


# REQUEST / RESPONSE SCHEMAS

class LeadRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    company_age_years: float = Field(..., ge=0)
    avg_review_score: float = Field(..., ge=0, le=5)
    review_count: int = Field(..., ge=0)

    company_status: str = "unknown"
    company_category: str = "unknown"
    industry: str = "unknown"
    company_size: str = "unknown"
    country: str = "unknown"
    location: str = "unknown"
    domain_type: str = "unknown"
    source_dataset: str = "unknown"

    contact_email: Optional[str] = None
    website_url: Optional[str] = None
    message_text: str = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    lead_log_id: int
    actual_quality_label: Optional[str] = None   # "good_lead" | "bad_lead" | "spam"
    actual_spam_label: Optional[str] = None       # "spam" | "ham"
    notes: Optional[str] = None


# GLOBALS

xgb_model = None
feature_columns = None
bert_tokenizer = None
bert_model = None
roberta_tokenizer = None
roberta_model = None

_retrain_process: subprocess.Popen = None
_retrain_lock = threading.Lock()


# DATABASE

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS scored_leads (
            lead_log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            company_name    TEXT,
            contact_email   TEXT,
            website_url     TEXT,
            message_text    TEXT,
            company_age_years   REAL,
            avg_review_score    REAL,
            review_count        INTEGER,
            company_status      TEXT,
            company_category    TEXT,
            industry            TEXT,
            company_size        TEXT,
            country             TEXT,
            domain_type         TEXT,
            predicted_score         REAL,
            quality_label           TEXT,
            spam_risk               TEXT,
            recommended_action      TEXT,
            bert_prob_spam          REAL,
            roberta_prob_spam       REAL,
            spam_score_rule         REAL,
            risk_score_rule         REAL,
            domain_trust_score      REAL,
            smtp_mailbox_accepted   INTEGER,
            is_disposable_provider  INTEGER,
            reasons                 TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_log_id         INTEGER NOT NULL REFERENCES scored_leads(lead_log_id),
            timestamp           TEXT NOT NULL,
            actual_quality_label    TEXT,
            actual_spam_label       TEXT,
            notes                   TEXT
        );
    """)
    con.commit()
    con.close()


def get_feedback_count() -> int:
    """Return total number of feedback rows in the database."""
    if not os.path.exists(DB_PATH):
        return 0
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    count = cur.fetchone()[0]
    con.close()
    return count


def load_retrain_state() -> dict:
    if not os.path.exists(RETRAIN_STATE_PATH):
        return {"last_retrain_feedback_count": 0, "last_retrain_at": None, "last_retrain_status": None}
    with open(RETRAIN_STATE_PATH, "r") as f:
        return json.load(f)


def save_retrain_state(state: dict):
    with open(RETRAIN_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def check_retrain_threshold():
    """After each feedback submission, check if enough new feedback has accumulated."""
    total = get_feedback_count()
    state = load_retrain_state()
    new_since_last = total - state.get("last_retrain_feedback_count", 0)
    if new_since_last >= AUTO_RETRAIN_THRESHOLD:
        print(
            f"\n{'='*60}\n"
            f"  [RETRAIN ALERT] {new_since_last} new feedback rows collected.\n"
            f"  Threshold of {AUTO_RETRAIN_THRESHOLD} reached.\n"
            f"  Run:  python src/retrain_from_feedback.py\n"
            f"  Or call POST /retrain/trigger via the API.\n"
            f"{'='*60}\n"
        )


def save_scored_lead(
    req,
    predicted_score: float,
    quality_label: str,
    spam_risk: str,
    recommended_action: str,
    bert_prob_spam: float,
    roberta_prob_spam: float,
    spam_score_rule: float,
    risk_score_rule: float,
    domain_trust_score: float,
    smtp_mailbox_accepted: int,
    is_disposable_provider: int,
    reasons: list,
) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO scored_leads (
            timestamp, company_name, contact_email, website_url, message_text,
            company_age_years, avg_review_score, review_count, company_status,
            company_category, industry, company_size, country, domain_type,
            predicted_score, quality_label, spam_risk, recommended_action,
            bert_prob_spam, roberta_prob_spam, spam_score_rule, risk_score_rule,
            domain_trust_score, smtp_mailbox_accepted, is_disposable_provider, reasons
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        req.company_name, req.contact_email, req.website_url, req.message_text,
        req.company_age_years, req.avg_review_score, req.review_count,
        req.company_status, req.company_category, req.industry,
        req.company_size, req.country, req.domain_type,
        predicted_score, quality_label, spam_risk, recommended_action,
        bert_prob_spam, roberta_prob_spam, spam_score_rule, risk_score_rule,
        domain_trust_score, smtp_mailbox_accepted, is_disposable_provider,
        json.dumps(reasons),
    ))
    lead_log_id = cur.lastrowid
    con.commit()
    con.close()
    return lead_log_id


# HELPERS

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_required_paths():
    required_paths = [
        XGB_MODEL_PATH,
        FEATURE_COLUMNS_PATH,
        BERT_MODEL_DIR,
        BERT_TOKENIZER_DIR,
        ROBERTA_MODEL_DIR,
        ROBERTA_TOKENIZER_DIR,
        STATIC_DIR,
        INDEX_HTML_PATH,
    ]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required model/artifact paths: {missing}")


def clip_score_01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def to_score_100(score_01: float) -> float:
    return float(np.clip(score_01 * 100.0, 0.0, 100.0))


def score_band(score_100: float) -> str:
    if score_100 >= 80:
        return "High quality"
    if score_100 >= 60:
        return "Promising"
    if score_100 >= 40:
        return "Needs review"
    return "Low quality"


def spam_risk_band(score_100: float) -> str:
    if score_100 >= 80:
        return "High"
    if score_100 >= 50:
        return "Medium"
    return "Low"


def recommended_action_from_score(score_100: float) -> str:
    if score_100 >= 80:
        return "Send to sales quickly"
    if score_100 >= 60:
        return "Review soon and prioritise"
    if score_100 >= 40:
        return "Review before contacting"
    return "Hold or deprioritise"


def model_message_assessment(score_01: float) -> str:
    if score_01 < 0.30:
        return "Normal message"
    if score_01 < 0.70:
        return "Somewhat suspicious"
    return "Highly suspicious"


# FEATURE EXTRACTION

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+\-']+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")

URGENT_TERMS = [
    "urgent", "immediately", "act now", "limited offer", "limited time",
    "asap", "now", "click here", "verify", "confirm", "final notice",
    "response needed", "exclusive offer", "won", "winner", "claim",
]

SPAM_TERMS = [
    "free", "offer", "winner", "guaranteed", "click", "act now", "claim",
    "prize", "limited", "urgent", "verify", "confirm", "bonus",
]


def extract_message_features(message_text: str) -> Dict[str, Any]:
    text = message_text.strip()
    text_lower = text.lower()

    has_url = int(bool(URL_PATTERN.search(text)))
    has_email = int(bool(EMAIL_PATTERN.search(text)))
    has_phone = int(bool(PHONE_PATTERN.search(text)))

    message_length_chars = len(text)
    token_count = max(1, len(text.split()))

    urgent_hits = sum(term in text_lower for term in URGENT_TERMS)
    urgency_score = min(1.0, urgent_hits / 4.0)

    needs_sliding_window = 1 if token_count > BASE_MAX_LENGTH else 0
    too_long_for_single_512_pass = 1 if token_count > 512 else 0

    spam_hits = sum(term in text_lower for term in SPAM_TERMS)

    return {
        "message_length_chars": message_length_chars,
        "token_count": token_count,
        "has_url": has_url,
        "has_email": has_email,
        "has_phone": has_phone,
        "urgency_score": urgency_score,
        "needs_sliding_window": needs_sliding_window,
        "too_long_for_single_512_pass": too_long_for_single_512_pass,
        "spam_hits": spam_hits,
    }


def compute_rule_scores(
    message_features: Dict[str, Any],
    company_age_years: float,
    avg_review_score: float,
    review_count: int,
    domain_type: str,
) -> tuple[float, float]:
    spam_score = 0.0
    risk_score = 0.0

    spam_score += message_features["has_url"] * 0.20
    spam_score += message_features["has_email"] * 0.05
    spam_score += message_features["has_phone"] * 0.08
    spam_score += message_features["urgency_score"] * 0.35
    spam_score += min(0.20, message_features["spam_hits"] * 0.05)

    if message_features["message_length_chars"] < 25:
        spam_score += 0.08
    if message_features["message_length_chars"] > 1200:
        spam_score += 0.10

    risk_score += message_features["has_url"] * 0.15
    risk_score += message_features["urgency_score"] * 0.25

    if company_age_years < 1:
        risk_score += 0.20
    elif company_age_years < 3:
        risk_score += 0.10

    if review_count == 0:
        risk_score += 0.20
    elif review_count < 5:
        risk_score += 0.10

    if avg_review_score < 2.5:
        risk_score += 0.20
    elif avg_review_score < 3.5:
        risk_score += 0.10

    if domain_type == "free":
        risk_score += 0.15
    elif domain_type == "unknown":
        risk_score += 0.08

    spam_score = float(np.clip(spam_score, 0.0, 1.0))
    risk_score = float(np.clip(risk_score, 0.0, 1.0))
    return spam_score, risk_score


# NLP INFERENCE

def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def build_chunk_inputs(text: str, tokenizer, max_length: int, stride: int):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_len = len(token_ids)

    use_sliding = token_len > max_length

    if not use_sliding:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "num_chunks": 1,
            "tokenized_length": token_len,
            "used_sliding_window": 0,
        }

    special_tokens_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_max_len = max_length - special_tokens_count

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    if cls_id is None or sep_id is None or pad_id is None:
        raise ValueError("Tokenizer is missing cls_token_id, sep_token_id, or pad_token_id.")

    chunk_input_ids = []
    chunk_attention_masks = []

    start = 0
    while start < token_len:
        end = min(start + content_max_len, token_len)
        chunk_token_ids = token_ids[start:end]

        full_ids = [cls_id] + chunk_token_ids + [sep_id]
        attention_mask = [1] * len(full_ids)

        pad_len = max_length - len(full_ids)
        if pad_len < 0:
            full_ids = full_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        else:
            full_ids = full_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        chunk_input_ids.append(full_ids)
        chunk_attention_masks.append(attention_mask)

        if end >= token_len:
            break
        start += stride

    return {
        "input_ids": torch.tensor(chunk_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(chunk_attention_masks, dtype=torch.long),
        "num_chunks": len(chunk_input_ids),
        "tokenized_length": token_len,
        "used_sliding_window": 1,
    }


@torch.no_grad()
def run_text_model(text: str, tokenizer, model):
    chunked = build_chunk_inputs(
        text=text,
        tokenizer=tokenizer,
        max_length=BASE_MAX_LENGTH,
        stride=SLIDING_WINDOW_STRIDE,
    )

    input_ids = chunked["input_ids"].to(DEVICE)
    attention_mask = chunked["attention_mask"].to(DEVICE)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    pooled_chunks = mean_pool_last_hidden(outputs.hidden_states[-1], attention_mask)

    aggregated_probs = probs.mean(dim=0, keepdim=True)
    aggregated_probs = torch.clamp(aggregated_probs, min=1e-6, max=1.0)
    aggregated_embedding = pooled_chunks.mean(dim=0, keepdim=True)

    pred_label_id = int(torch.argmax(aggregated_probs, dim=1).cpu().item())

    return {
        "pred_label_id": pred_label_id,
        "prob_ham": float(aggregated_probs[0, 0].cpu().item()),
        "prob_spam": float(aggregated_probs[0, 1].cpu().item()),
        "num_chunks": int(chunked["num_chunks"]),
        "tokenized_length": int(chunked["tokenized_length"]),
        "used_sliding_window": int(chunked["used_sliding_window"]),
        "embedding": aggregated_embedding.cpu().numpy()[0],
    }


# FEATURE VECTOR BUILDING

def set_if_exists(row: dict, key: str, value):
    if key in row:
        row[key] = value


def build_xgb_input_row(
    feature_columns,
    req: LeadRequest,
    message_features: Dict[str, Any],
    spam_score_rule: float,
    risk_score_rule: float,
    bert_result: Dict[str, Any],
    roberta_result: Dict[str, Any],
    email_domain_result: Dict[str, Any],
):
    row = {col: 0.0 for col in feature_columns}

    is_active_company = 1 if req.company_status == "active" else 0
    has_reviews = 1 if req.review_count > 0 else 0
    is_corporate_domain = 1 if req.domain_type == "corporate" else 0
    good_review_score_flag = 1 if req.avg_review_score >= 3.5 else 0
    company_age_3plus_flag = 1 if req.company_age_years >= 3 else 0

    numeric_map = {
        "company_age_years": req.company_age_years,
        "avg_review_score": req.avg_review_score,
        "review_count": req.review_count,
        "message_length_chars": message_features["message_length_chars"],
        "token_count": message_features["token_count"],
        "has_url": message_features["has_url"],
        "has_email": message_features["has_email"],
        "has_phone": message_features["has_phone"],
        "urgency_score": message_features["urgency_score"],
        "needs_sliding_window": message_features["needs_sliding_window"],
        "too_long_for_single_512_pass": message_features["too_long_for_single_512_pass"],
        "spam_score_rule": spam_score_rule,
        "risk_score_rule": risk_score_rule,
        "is_active_company": is_active_company,
        "has_reviews": has_reviews,
        "is_corporate_domain": is_corporate_domain,
        "good_review_score_flag": good_review_score_flag,
        "company_age_3plus_flag": company_age_3plus_flag,

        "bert_pred_label_id": bert_result["pred_label_id"],
        "bert_prob_ham": bert_result["prob_ham"],
        "bert_prob_spam": bert_result["prob_spam"],
        "bert_num_chunks": bert_result["num_chunks"],
        "bert_tokenized_length": bert_result["tokenized_length"],
        "bert_used_sliding_window": bert_result["used_sliding_window"],

        "roberta_pred_label_id": roberta_result["pred_label_id"],
        "roberta_prob_ham": roberta_result["prob_ham"],
        "roberta_prob_spam": roberta_result["prob_spam"],
        "roberta_num_chunks": roberta_result["num_chunks"],
        "roberta_tokenized_length": roberta_result["tokenized_length"],
        "roberta_used_sliding_window": roberta_result["used_sliding_window"],

        "email_syntax_valid": email_domain_result.get("email_syntax_valid", 0),
        "domain_present": email_domain_result.get("domain_present", 0),
        "domain_has_mx": email_domain_result.get("domain_has_mx", 0),
        "domain_has_a": email_domain_result.get("domain_has_a", 0),
        "smtp_reachable": email_domain_result.get("smtp_reachable", 0),
        "smtp_mailbox_accepted": email_domain_result.get("smtp_mailbox_accepted", 0),
        "is_free_provider": email_domain_result.get("is_free_provider", 0),
        "is_disposable_provider": email_domain_result.get("is_disposable_provider", 0),
        "domain_matches_company_name": email_domain_result.get("domain_matches_company_name", 0),
        "domain_trust_score": email_domain_result.get("domain_trust_score", 0.0),
        "smtp_response_code": float(email_domain_result["smtp_response_code"])
        if email_domain_result.get("smtp_response_code") is not None else 0.0,
    }

    for k, v in numeric_map.items():
        set_if_exists(row, k, v)

    for i, v in enumerate(bert_result["embedding"]):
        set_if_exists(row, f"bert_emb_{i}", float(v))

    for i, v in enumerate(roberta_result["embedding"]):
        set_if_exists(row, f"roberta_emb_{i}", float(v))

    categorical_flags = {
        f"company_status_{req.company_status}": 1,
        f"company_category_{req.company_category}": 1,
        f"industry_{req.industry}": 1,
        f"company_size_{req.company_size}": 1,
        f"country_{req.country}": 1,
        f"location_{req.location}": 1,
        f"domain_type_{req.domain_type}": 1,
        f"source_dataset_{req.source_dataset}": 1,
    }

    email_domain_categoricals = {
        f"domain_type_{email_domain_result.get('domain_type', 'unknown')}": 1,
        f"resolved_domain_{str(email_domain_result.get('resolved_domain') or 'unknown')}": 1,
        f"smtp_response_message_{str(email_domain_result.get('smtp_response_message') or 'unknown')}": 1,
        f"smtp_exception_{str(email_domain_result.get('smtp_exception') or 'unknown')}": 1,
        f"mx_hosts_{str(email_domain_result.get('mx_hosts') or 'unknown')}": 1,
        f"input_contact_email_{str(email_domain_result.get('input_contact_email') or 'unknown')}": 1,
        f"input_website_url_{str(email_domain_result.get('input_website_url') or 'unknown')}": 1,
        f"extracted_email_from_text_{str(email_domain_result.get('extracted_email_from_text') or 'unknown')}": 1,
        f"extracted_url_from_text_{str(email_domain_result.get('extracted_url_from_text') or 'unknown')}": 1,
        f"resolved_email_{str(email_domain_result.get('resolved_email') or 'unknown')}": 1,
    }

    for k, v in {**categorical_flags, **email_domain_categoricals}.items():
        set_if_exists(row, k, v)

    return pd.DataFrame([row])


def business_reasons(
    message_features,
    spam_score_rule,
    risk_score_rule,
    bert_result,
    roberta_result,
    req: LeadRequest,
    email_domain_result: Dict[str, Any],
):
    reasons = []

    if message_features["urgency_score"] >= 0.7:
        reasons.append("Message sounds urgent or pressuring")
    if message_features["has_url"] == 1:
        reasons.append("Message contains a link")
    if max(bert_result["prob_spam"], roberta_result["prob_spam"]) >= 0.75:
        reasons.append("Message wording resembles suspicious messages")
    if req.review_count == 0 or req.avg_review_score < 3.0:
        reasons.append("Company credibility signals are weak")
    if req.company_age_years < 2:
        reasons.append("Company appears relatively new")
    if req.domain_type == "free":
        reasons.append("Uses a free email or domain pattern")
    if spam_score_rule >= 0.70:
        reasons.append("Rules-based spam checks raised concern")
    if risk_score_rule >= 0.70:
        reasons.append("Rules-based risk checks raised concern")

    if email_domain_result.get("email_syntax_valid", 0) == 0:
        reasons.append("Email format looks invalid or missing")
    if email_domain_result.get("is_disposable_provider", 0) == 1:
        reasons.append("Email appears to use a disposable provider")
    if email_domain_result.get("smtp_mailbox_accepted", 0) == 0 and email_domain_result.get("resolved_email"):
        reasons.append("Mailbox could not be confirmed by SMTP check")
    if email_domain_result.get("domain_matches_company_name", 0) == 1:
        reasons.append("Email domain appears consistent with the company name")
    if float(email_domain_result.get("domain_trust_score", 0.0)) >= 0.7:
        reasons.append("Email and domain trust signals look stronger")

    final_reasons = []
    for r in reasons:
        if r not in final_reasons:
            final_reasons.append(r)

    return final_reasons[:5]


# SCHEDULER

def _seconds_until(target_time_str: str) -> float:
    """Return seconds until the next occurrence of HH:MM (local time)."""
    now = datetime.now()
    hour, minute = map(int, target_time_str.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _scheduled_retrain_loop():
    """
    Background daemon thread. Wakes up daily at SCHEDULED_RETRAIN_TIME,
    checks whether enough new feedback exists, and triggers retraining if so.
    """
    import time

    print(f"[Scheduler] Nightly retrain scheduled at {SCHEDULED_RETRAIN_TIME} "
          f"(min {SCHEDULED_RETRAIN_MIN_XGB} new feedback rows required).")

    while True:
        wait = _seconds_until(SCHEDULED_RETRAIN_TIME)
        next_run = datetime.now() + timedelta(seconds=wait)
        print(f"[Scheduler] Next check at {next_run.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(in {wait/3600:.1f} h)")

        time.sleep(wait)

        # Wake up — check feedback count
        total          = get_feedback_count()
        state          = load_retrain_state()
        new_since_last = total - state.get("last_retrain_feedback_count", 0)

        print(f"[Scheduler] Wake-up — {new_since_last} new feedback rows since last retrain.")

        if new_since_last < SCHEDULED_RETRAIN_MIN_XGB:
            print(f"[Scheduler] Skipping — need {SCHEDULED_RETRAIN_MIN_XGB}, "
                  f"have {new_since_last}.")
            continue

        # Check nothing is already running
        global _retrain_process
        with _retrain_lock:
            if _retrain_process is not None and _retrain_process.poll() is None:
                print("[Scheduler] Skipping — retraining already in progress.")
                continue

            if not os.path.exists(RETRAIN_SCRIPT_PATH):
                print(f"[Scheduler] ERROR — retrain script not found: {RETRAIN_SCRIPT_PATH}")
                continue

            log_path = os.path.join(PROJECT_ROOT, "outputs", "retrain_log.txt")
            log_file = open(log_path, "w")

            _retrain_process = subprocess.Popen(
                [sys.executable, RETRAIN_SCRIPT_PATH,
                 f"--min-nlp=50", f"--min-xgb={SCHEDULED_RETRAIN_MIN_XGB}"],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

            save_retrain_state({
                "last_retrain_feedback_count": total,
                "last_retrain_at":             datetime.now(timezone.utc).isoformat(),
                "last_retrain_status":         "running",
            })

            print(f"[Scheduler] Retraining started (PID {_retrain_process.pid}). "
                  f"Log: {log_path}")


# STARTUP / LIFESPAN

def load_all_models():
    global xgb_model, feature_columns
    global bert_tokenizer, bert_model
    global roberta_tokenizer, roberta_model

    check_required_paths()

    xgb_model = joblib.load(XGB_MODEL_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)["feature_columns"]

    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_DIR, use_fast=True)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_DIR,
        output_hidden_states=True,
    )
    bert_model.to(DEVICE)
    bert_model.eval()

    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_TOKENIZER_DIR, use_fast=True)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_DIR,
        output_hidden_states=True,
    )
    roberta_model.to(DEVICE)
    roberta_model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_all_models()
    if SCHEDULED_RETRAIN_ENABLED:
        t = threading.Thread(target=_scheduled_retrain_loop, daemon=True)
        t.start()
    yield


app = FastAPI(
    title="Lead Quality, Risk & Spam Detection API",
    version="1.0.0",
    description="Scores inbound business leads using rules, BERT, RoBERTa, email/domain validation, SMTP checks, and XGBoost.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ROUTES

@app.get("/")
def root():
    return FileResponse(INDEX_HTML_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": {
            "xgboost": xgb_model is not None,
            "bert": bert_model is not None,
            "roberta": roberta_model is not None,
        },
    }


@app.post("/score")
def score_lead(req: LeadRequest):
    try:
        if xgb_model is None or bert_model is None or roberta_model is None:
            raise HTTPException(status_code=500, detail="Models are not loaded.")

        message_features = extract_message_features(req.message_text)

        email_domain_result = build_email_domain_features(
            company_name=req.company_name,
            contact_email=req.contact_email,
            website_url=req.website_url,
            message_text=req.message_text,
            enable_smtp_check=True,
        )

        effective_domain_type = email_domain_result.get("domain_type") or req.domain_type

        spam_score_rule, risk_score_rule = compute_rule_scores(
            message_features=message_features,
            company_age_years=req.company_age_years,
            avg_review_score=req.avg_review_score,
            review_count=req.review_count,
            domain_type=effective_domain_type,
        )

        bert_result = run_text_model(req.message_text, bert_tokenizer, bert_model)
        roberta_result = run_text_model(req.message_text, roberta_tokenizer, roberta_model)

        X_input = build_xgb_input_row(
            feature_columns=feature_columns,
            req=req,
            message_features=message_features,
            spam_score_rule=spam_score_rule,
            risk_score_rule=risk_score_rule,
            bert_result=bert_result,
            roberta_result=roberta_result,
            email_domain_result=email_domain_result,
        )

        pred_score_raw = float(xgb_model.predict(X_input)[0])
        pred_score_raw = clip_score_01(pred_score_raw)
        pred_score = to_score_100(pred_score_raw)

        quality_label = score_band(pred_score)

        spam_risk_signal_100 = max(
            spam_score_rule * 100.0,
            risk_score_rule * 100.0,
            bert_result["prob_spam"] * 100.0,
            roberta_result["prob_spam"] * 100.0,
            (1.0 - float(email_domain_result.get("domain_trust_score", 0.0))) * 100.0,
        )
        spam_risk = spam_risk_band(spam_risk_signal_100)
        action = recommended_action_from_score(pred_score)

        reasons = business_reasons(
            message_features=message_features,
            spam_score_rule=spam_score_rule,
            risk_score_rule=risk_score_rule,
            bert_result=bert_result,
            roberta_result=roberta_result,
            req=req,
            email_domain_result=email_domain_result,
        )

        lead_log_id = save_scored_lead(
            req=req,
            predicted_score=round(pred_score, 1),
            quality_label=quality_label,
            spam_risk=spam_risk,
            recommended_action=action,
            bert_prob_spam=bert_result["prob_spam"],
            roberta_prob_spam=roberta_result["prob_spam"],
            spam_score_rule=spam_score_rule,
            risk_score_rule=risk_score_rule,
            domain_trust_score=float(email_domain_result.get("domain_trust_score", 0.0)),
            smtp_mailbox_accepted=int(email_domain_result.get("smtp_mailbox_accepted", 0)),
            is_disposable_provider=int(email_domain_result.get("is_disposable_provider", 0)),
            reasons=reasons,
        )

        return {
            "lead_log_id": lead_log_id,
            "company_name": req.company_name,
            "overall_score": round(pred_score, 1),
            "lead_quality": quality_label,
            "spam_risk": spam_risk,
            "recommended_action": action,
            "why_this_lead_was_scored_this_way": reasons,
            "automatic_checks": {
                "urgency_level": round(message_features["urgency_score"], 2),
                "rules_based_spam_score": int(round(spam_score_rule * 100.0, 0)),
                "rules_based_risk_score": int(round(risk_score_rule * 100.0, 0)),
                "bert_suspicious_message_score_pct": round(bert_result["prob_spam"] * 100.0, 1),
                "roberta_suspicious_message_score_pct": round(roberta_result["prob_spam"] * 100.0, 1),
                "bert_message_assessment": model_message_assessment(bert_result["prob_spam"]),
                "roberta_message_assessment": model_message_assessment(roberta_result["prob_spam"]),
                "contains_link": bool(message_features["has_url"]),
                "contains_email": bool(message_features["has_email"]),
                "contains_phone": bool(message_features["has_phone"]),
                "message_length": message_features["message_length_chars"],
                "token_count": message_features["token_count"],

                "resolved_email": email_domain_result.get("resolved_email"),
                "resolved_domain": email_domain_result.get("resolved_domain"),
                "email_syntax_valid": bool(email_domain_result.get("email_syntax_valid", 0)),
                "domain_present": bool(email_domain_result.get("domain_present", 0)),
                "domain_has_mx": bool(email_domain_result.get("domain_has_mx", 0)),
                "domain_has_a": bool(email_domain_result.get("domain_has_a", 0)),
                "smtp_reachable": bool(email_domain_result.get("smtp_reachable", 0)),
                "smtp_mailbox_accepted": bool(email_domain_result.get("smtp_mailbox_accepted", 0)),
                "is_free_provider": bool(email_domain_result.get("is_free_provider", 0)),
                "is_disposable_provider": bool(email_domain_result.get("is_disposable_provider", 0)),
                "domain_matches_company_name": bool(email_domain_result.get("domain_matches_company_name", 0)),
                "domain_type_detected": email_domain_result.get("domain_type"),
                "domain_trust_score": email_domain_result.get("domain_trust_score"),
                "smtp_response_code": email_domain_result.get("smtp_response_code"),
                "smtp_response_message": email_domain_result.get("smtp_response_message"),
                "smtp_exception": email_domain_result.get("smtp_exception"),
                "mx_hosts": email_domain_result.get("mx_hosts"),
            },
            "model_debug": {
                "xgboost_raw_score_0_to_1": round(pred_score_raw, 6),
                "xgboost_display_score_0_to_100": round(pred_score, 2),
                "bert_prob_spam": round(bert_result["prob_spam"], 6),
                "roberta_prob_spam": round(roberta_result["prob_spam"], 6),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("SELECT lead_log_id FROM scored_leads WHERE lead_log_id = ?", (req.lead_log_id,))
    if cur.fetchone() is None:
        con.close()
        raise HTTPException(status_code=404, detail=f"lead_log_id {req.lead_log_id} not found.")

    cur.execute("""
        INSERT INTO feedback (lead_log_id, timestamp, actual_quality_label, actual_spam_label, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (
        req.lead_log_id,
        datetime.now(timezone.utc).isoformat(),
        req.actual_quality_label,
        req.actual_spam_label,
        req.notes,
    ))
    con.commit()
    con.close()

    check_retrain_threshold()

    return {"status": "ok", "message": "Feedback recorded. Thank you."}


@app.get("/feedback/{lead_log_id}")
def get_feedback(lead_log_id: int):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT * FROM scored_leads WHERE lead_log_id = ?", (lead_log_id,))
    lead = cur.fetchone()
    if lead is None:
        con.close()
        raise HTTPException(status_code=404, detail=f"lead_log_id {lead_log_id} not found.")

    cur.execute("SELECT * FROM feedback WHERE lead_log_id = ?", (lead_log_id,))
    feedback_rows = [dict(row) for row in cur.fetchall()]
    con.close()

    return {
        "lead": dict(lead),
        "feedback": feedback_rows,
    }


@app.post("/retrain/trigger")
def trigger_retrain(min_nlp: int = 50, min_xgb: int = 20):
    """
    Launch the retraining script as a background subprocess.
    The API stays live during retraining. Restart the server once retraining completes
    to load the updated models.
    """
    global _retrain_process

    with _retrain_lock:
        # Check if a retrain is already running
        if _retrain_process is not None and _retrain_process.poll() is None:
            return {
                "status": "already_running",
                "message": "Retraining is already in progress.",
                "pid": _retrain_process.pid,
            }

        if not os.path.exists(RETRAIN_SCRIPT_PATH):
            raise HTTPException(status_code=500, detail=f"Retrain script not found: {RETRAIN_SCRIPT_PATH}")

        # Check enough feedback exists before launching
        total = get_feedback_count()
        state = load_retrain_state()
        new_since_last = total - state.get("last_retrain_feedback_count", 0)

        if new_since_last < min_xgb:
            return {
                "status": "insufficient_feedback",
                "message": f"Need at least {min_xgb} new feedback rows. Have {new_since_last} since last retrain.",
                "total_feedback": total,
                "new_since_last_retrain": new_since_last,
            }

        cmd = [sys.executable, RETRAIN_SCRIPT_PATH,
               f"--min-nlp={min_nlp}", f"--min-xgb={min_xgb}"]

        log_path = os.path.join(PROJECT_ROOT, "outputs", "retrain_log.txt")
        log_file = open(log_path, "w")

        _retrain_process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        # Update state immediately so threshold counter resets
        save_retrain_state({
            "last_retrain_feedback_count": total,
            "last_retrain_at": datetime.now(timezone.utc).isoformat(),
            "last_retrain_status": "running",
        })

        print(f"[Retrain] Started retraining subprocess (PID {_retrain_process.pid}). "
              f"Log: {log_path}")

        return {
            "status": "started",
            "message": "Retraining has started in the background. "
                       "Restart the API server once it completes to load updated models.",
            "pid": _retrain_process.pid,
            "log_file": log_path,
        }


@app.get("/retrain/status")
def retrain_status():
    """Return the current retraining status and feedback counts."""
    global _retrain_process

    total = get_feedback_count()
    state = load_retrain_state()
    new_since_last = total - state.get("last_retrain_feedback_count", 0)

    process_status = "idle"
    pid = None

    if _retrain_process is not None:
        poll = _retrain_process.poll()
        if poll is None:
            process_status = "running"
            pid = _retrain_process.pid
        elif poll == 0:
            process_status = "completed"
        else:
            process_status = f"failed (exit code {poll})"

    return {
        "process_status":          process_status,
        "pid":                     pid,
        "total_feedback_rows":     total,
        "new_since_last_retrain":  new_since_last,
        "auto_retrain_threshold":  AUTO_RETRAIN_THRESHOLD,
        "threshold_reached":       new_since_last >= AUTO_RETRAIN_THRESHOLD,
        "last_retrain_at":         state.get("last_retrain_at"),
        "last_retrain_status":     state.get("last_retrain_status"),
    }


# LOCAL RUN

if __name__ == "__main__":
    import uvicorn

    if AUTO_OPEN_BROWSER:
        webbrowser.open("http://127.0.0.1:8000/")

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)