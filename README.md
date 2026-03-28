# AI-Powered Lead and Contact Form Quality Scoring System

A production-ready machine learning system, developed in collaboration with [Palash Kumar](https://github.com/palash-kumar), that automatically scores the quality of inbound business leads and contact form submissions. It cascades two fine-tuned transformer models (BERT and RoBERTa) with an XGBoost regressor, live email/DNS validation, and SHAP-based explainability into a single REST API with a polished browser UI.
---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Feedback and Retraining](#feedback-and-retraining)
- [Dataset](#dataset)
- [Notebooks](#notebooks)

---

## Overview

Sales and marketing teams receive hundreds of inbound leads daily through contact forms, ranging from high-value prospects to automated spam and phishing attempts. Manual triage is slow, inconsistent, and does not scale.

This system solves that by producing a continuous quality score (0-100) for every submission, a recommended action, and a plain-English explanation of what drove the score, all in under 500 ms.

**What it does at inference time:**

1. Runs the message text through fine-tuned BERT and RoBERTa classifiers to obtain spam probability scores and 768-dimensional contextual embeddings.
2. Validates the contact email and domain in real time via DNS (MX/A records) and an optional SMTP handshake.
3. Combines transformer outputs, email validation signals, and structured company attributes into a 4,137-feature vector.
4. Feeds that vector to an XGBoost regressor to predict the lead quality score.
5. Decomposes the prediction with SHAP and returns a human-readable explanation alongside the score.

---

## Architecture

```
Contact Form Submission
        |
        v
+-------+--------+     +------------------+
|  Message Text  |---->|  BERT Classifier  |----> P(spam), 768-dim embedding
+----------------+     +------------------+
        |
        |              +--------------------+
        +------------->| RoBERTa Classifier |----> P(spam), 768-dim embedding
                       +--------------------+
                                |
+----------------+              |
| Company Fields |              |
| (age, status,  |              v
| industry, size)|    +---------------------+
+----------------+--->|   XGBoost Regressor |----> Quality Score (0-100)
        |              |   (4,137 features)  |           |
+----------------+     +---------------------+           v
| Email/Domain   |                              Recommended Action
| Validation     |                              SHAP Explanation
| (DNS + SMTP)   |
+----------------+
```

**Sliding-window tokenization** (window=256, stride=192) handles messages longer than the 512-token BERT limit without any architectural modification. The highest-confidence chunk embedding is used as the representative embedding for the full message.

**Score-to-action mapping:**

| Quality Score | Recommended Action  |
|:-------------:|---------------------|
| 70 - 100      | Send to Sales       |
| 40 - 69       | Needs Review        |
| 20 - 39       | Block or Quarantine |
| 0 - 19        | Manual Risk Review  |

---

## Performance

### NLP Models (test set, n = 6,055)

| Model                   | Accuracy  | Macro-F1  | Precision | Recall    |
|-------------------------|:---------:|:---------:|:---------:|:---------:|
| BERT (bert-base-uncased)| **99.12%**| **99.10%**| 99.52%    | 98.68%    |
| RoBERTa (roberta-base)  | **99.41%**| **99.39%**| 99.56%    | 99.22%    |

### XGBoost Regressor (test set, n = 4,500)

| Metric | Train     | Validation | Test      |
|--------|:---------:|:----------:|:---------:|
| MAE    | 0.0005    | 0.0011     | **0.0011**|
| RMSE   | 0.0012    | 0.0031     | **0.0031**|
| R²     | 0.9999    | 0.9996     | **0.9996**|

### Top Feature Importances (XGBoost, gain)

| Rank | Feature                    | Importance | Category              |
|------|----------------------------|:----------:|-----------------------|
| 1    | spam_score_rule            | 32.7%      | Rule-based signal     |
| 2    | company_age_3plus_flag     | 12.9%      | Structured            |
| 3    | company_age_years          | 12.6%      | Structured            |
| 4    | roberta_emb_120            | 5.6%       | Transformer embedding |
| 5    | roberta_emb_688            | 5.6%       | Transformer embedding |
| 6    | company_status_Active      | 5.4%       | Structured            |

---

## Project Structure

```
.
├── api/
│   └── main.py                  # FastAPI application (scoring, feedback, retraining endpoints)
├── src/
│   ├── build_lead_dataset.py    # Phase 1: merge company records with message corpora
│   ├── build_phase2_datasets.py # Phase 2: 70/15/15 train/val/test splits
│   ├── email_domain_features.py # Email validation core (DNS, SMTP, disposable detection)
│   ├── generate_email_domain_features.py  # Batch email feature generation
│   ├── generate_hybrid_nlp_features.py    # Batch BERT/RoBERTa embedding extraction
│   ├── generate_xgboost_business_explanations.py  # SHAP explanation generation
│   ├── train_bert_classifier.py           # Fine-tune BERT for spam/ham
│   ├── train_roberta_classifier.py        # Fine-tune RoBERTa for spam/ham
│   ├── train_xgboost_hybrid.py            # Train XGBoost hybrid regressor
│   └── retrain_from_feedback.py           # Incremental retraining from user feedback
├── notebooks/
│   ├── 01_data_checks.ipynb               # Data validation and statistics
│   ├── 02_phase2_dataset_checks.ipynb     # Split verification and class balance
│   ├── 03_bert_visualisations.ipynb       # BERT metrics, confusion matrix, t-SNE
│   └── 04_roberta_visualisations.ipynb    # RoBERTa metrics and comparative analysis
├── static/
│   ├── index.html                         # Single-page application (Tailwind CSS + jQuery)
│   └── favicon.svg
├── outputs/
│   ├── phase2/                            # Train/val/test splits
│   └── phase3/
│       ├── bert/                          # BERT checkpoint, tokenizer, metrics, embeddings
│       ├── roberta/                       # RoBERTa checkpoint, tokenizer, metrics, embeddings
│       ├── xgboost/                       # XGBoost model, feature_columns.json, SHAP values
│       ├── nlp_features/                  # Pre-computed 768-dim embeddings (train/val/test)
│       └── email_domain_features/         # Pre-computed email validation features
├── Dataset/                               # Raw source datasets (not tracked in git)
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer              | Technology                                             |
|--------------------|--------------------------------------------------------|
| Web framework      | FastAPI + Uvicorn                                      |
| NLP models         | Hugging Face Transformers (bert-base-uncased, roberta-base) |
| Deep learning      | PyTorch                                                |
| Gradient boosting  | XGBoost                                                |
| Email validation   | dnspython (DNS), smtplib (SMTP)                        |
| Explainability     | SHAP (TreeSHAP), Partial Dependence Plots              |
| Feedback storage   | SQLite                                                 |
| Frontend           | Tailwind CSS, jQuery, vanilla JS                       |
| Analytics UI       | Streamlit                                              |
| Model serialisation| Hugging Face safetensors, joblib                       |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A virtual environment (recommended)
- Trained model artifacts in `outputs/phase3/` (see [Training](#training) below)

### Installation

```bash
git clone https://github.com/your-username/lead-quality-scoring.git
cd lead-quality-scoring

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the API

```bash
uvicorn api.main:app --reload
```

The API starts at `http://localhost:8000`. Open that URL in a browser to load the scoring UI.

Interactive API docs are available at `http://localhost:8000/docs`.

### Training (optional, if you want to retrain from scratch)

Run each step in order:

```bash
# Phase 1 - build the lead dataset
python src/build_lead_dataset.py

# Phase 2 - create train/val/test splits
python src/build_phase2_datasets.py

# Phase 3 - train models
python src/train_bert_classifier.py
python src/train_roberta_classifier.py
python src/generate_hybrid_nlp_features.py
python src/generate_email_domain_features.py
python src/train_xgboost_hybrid.py

# Optional - generate SHAP business explanations
python src/generate_xgboost_business_explanations.py
```

---

## API Reference

### `POST /score`

Score an inbound lead. Returns a quality score, recommended action, SHAP-driven reasons, and a log ID for feedback submission.

**Request body:**

```json
{
  "company_name": "Acme Ltd",
  "company_age_years": 8.5,
  "avg_review_score": 4.2,
  "review_count": 312,
  "company_status": "Active",
  "company_category": "Private Limited Company",
  "industry": "Information Technology",
  "company_size": "51-200",
  "country": "United Kingdom",
  "location": "London",
  "domain_type": "corporate",
  "source_dataset": "companies_house",
  "contact_email": "john.doe@acme.co.uk",
  "website_url": "https://acme.co.uk",
  "message_text": "Hi, we are interested in your enterprise plan..."
}
```

**Response:**

```json
{
  "predicted_score": 84.3,
  "quality_label": "High quality",
  "spam_risk": "Low",
  "recommended_action": "Send to Sales",
  "bert_spam_prob": 0.021,
  "roberta_spam_prob": 0.018,
  "email_valid": true,
  "domain_has_mx": true,
  "smtp_reachable": true,
  "reasons": ["Active company with 8.5 years trading history", "..."],
  "lead_log_id": 42
}
```

### `POST /feedback`

Submit a correction label for a previously scored lead. Triggers automatic retraining once the feedback threshold is reached.

```json
{
  "lead_log_id": 42,
  "actual_quality_label": "good_lead",
  "actual_spam_label": "ham",
  "notes": "Converted to paying customer within 2 weeks"
}
```

### `GET /retrain/status`

Returns current retraining state: whether a process is running, total feedback count, and distance to the next automatic trigger.

### `POST /retrain/trigger`

Manually trigger retraining. Spawns `src/retrain_from_feedback.py` as a background subprocess so the API stays live.

### `GET /health`

Health check. Returns model load status and active compute device (CPU/CUDA).

---

## Feedback and Retraining

The system includes a closed-loop feedback mechanism:

1. After scoring a lead, the UI shows a feedback bar ("Was this assessment correct?").
2. The operator submits a correction label (`good_lead`, `bad_lead`, or `spam`) and optional notes.
3. Feedback is persisted to `outputs/feedback.db` (SQLite, two tables: `scored_leads` and `feedback`).
4. Once 100 new feedback rows accumulate, retraining triggers automatically.
5. A background daemon also runs nightly at 02:00 and retrains if at least 20 new feedback rows exist since the last run.
6. Retraining fine-tunes BERT and RoBERTa on the feedback texts, then rebuilds the XGBoost regressor using original training data merged with feedback-derived feature rows.
7. Before overwriting any checkpoint, the system creates a timestamped backup (up to 3 backups retained).

Retraining runs as a background subprocess, so the API continues serving requests throughout.

```bash
# Manually run retraining from the command line
python src/retrain_from_feedback.py --min-nlp 50 --min-xgb 20

# Dry run (shows what would happen without saving anything)
python src/retrain_from_feedback.py --dry-run
```

---

## Dataset

The training dataset was constructed synthetically by pairing company records with messages from four publicly available text corpora.

### Company Records

Sourced from the **UK Companies House** public bulk data product, containing registration details for companies incorporated under UK law. Fields include company status, SIC industry code, size band, age, and location.

### Message Corpora

| Source                        | Messages | Labels       |
|-------------------------------|:--------:|--------------|
| Enron Email Corpus            | 20,376   | Ham and Spam |
| SpamAssassin Public Corpus    | 4,589    | Ham and Spam |
| SMS Spam Collection (UCI)     | 2,526    | Ham and Spam |
| Additional ham corpus         | 2,509    | Ham          |
| **Total**                     | **40,364** | **51.4% Ham / 48.6% Spam** |

### Splits

| Split      | NLP rows | Structured rows |
|------------|:--------:|:---------------:|
| Train      | 28,254   | 21,000          |
| Validation | 6,055    | 4,500           |
| Test       | 6,055    | 4,500           |

The quality target (`quality_score_rule`) is a continuous value in [0, 1] derived from company maturity signals (active status, age, review score) and the message spam label.

---

## Notebooks

| Notebook                           | Description                                               |
|------------------------------------|-----------------------------------------------------------|
| `01_data_checks.ipynb`             | Row counts, missing values, distribution of company age, review scores, and message lengths |
| `02_phase2_dataset_checks.ipynb`   | Split verification, class balance, and feature statistics across train/val/test |
| `03_bert_visualisations.ipynb`     | BERT confusion matrix, ROC curve, F1 by class, t-SNE of embeddings |
| `04_roberta_visualisations.ipynb`  | RoBERTa equivalent visualisations and comparative analysis against BERT |

---

## License

This project is released for academic and research purposes. Do cite the author.
Name: Sheikh Khairul Momin Mohammad Tahmid
Email: sheikh.k.m.m.tahmid@gmail.com
Date of Completion: 24th March 2026
