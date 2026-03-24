import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# CONFIG

BATCH_SIZE = 16
BASE_MAX_LENGTH = 256
SLIDING_WINDOW_STRIDE = 192

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PHASE2_DIR = os.path.join(OUTPUTS_DIR, "phase2")
PHASE3_DIR = os.path.join(OUTPUTS_DIR, "phase3")

# Structured splits
TRAIN_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "train_structured.csv")
VAL_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "val_structured.csv")
TEST_STRUCTURED_PATH = os.path.join(PHASE2_DIR, "test_structured.csv")

# Saved BERT artifacts
BERT_MODEL_DIR = os.path.join(PHASE3_DIR, "bert", "model")
BERT_TOKENIZER_DIR = os.path.join(PHASE3_DIR, "bert", "tokenizer")

# Saved RoBERTa artifacts
ROBERTA_MODEL_DIR = os.path.join(PHASE3_DIR, "roberta", "model")
ROBERTA_TOKENIZER_DIR = os.path.join(PHASE3_DIR, "roberta", "tokenizer")

# Output
NLP_FEATURES_DIR = os.path.join(PHASE3_DIR, "nlp_features")
TRAIN_OUT = os.path.join(NLP_FEATURES_DIR, "train_nlp_features.csv")
VAL_OUT = os.path.join(NLP_FEATURES_DIR, "val_nlp_features.csv")
TEST_OUT = os.path.join(NLP_FEATURES_DIR, "test_nlp_features.csv")
SUMMARY_OUT = os.path.join(NLP_FEATURES_DIR, "hybrid_nlp_features_summary.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


# DATA

def load_structured_split(path: str, split_name: str) -> pd.DataFrame:
    check_file_exists(path)
    df = pd.read_csv(path)

    required_cols = [
        "lead_id",
        "message_id",
        "message_text",
    ]
    check_required_columns(df, required_cols, split_name)

    df["lead_id"] = pd.to_numeric(df["lead_id"], errors="coerce").astype(int)
    df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce").astype(int)
    df["message_text"] = df["message_text"].fillna("").astype(str).str.strip()
    df = df[df["message_text"] != ""].copy()

    return df.reset_index(drop=True)


class StructuredTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "lead_id": int(row["lead_id"]),
            "message_id": int(row["message_id"]),
            "text": str(row["message_text"]),
        }


def collate_samples(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


# TOKENIZATION WITH SELECTIVE SLIDING WINDOW

def build_chunk_inputs(
    text: str,
    tokenizer,
    max_length: int,
    stride: int,
) -> Dict[str, Any]:
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

    if content_max_len <= 0:
        raise ValueError("Invalid max_length for tokenizer special tokens.")

    if stride >= content_max_len:
        raise ValueError(
            f"sliding_window_stride ({stride}) must be smaller than content window size ({content_max_len})."
        )

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

    input_ids_tensor = torch.tensor(chunk_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(chunk_attention_masks, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "num_chunks": input_ids_tensor.size(0),
        "tokenized_length": token_len,
        "used_sliding_window": 1,
    }


# MODEL INFERENCE

@torch.no_grad()
def forward_one_sample(
    model,
    tokenizer,
    text: str,
    max_length: int,
    stride: int,
    device: torch.device,
) -> Dict[str, Any]:
    chunked = build_chunk_inputs(
        text=text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    input_ids = chunked["input_ids"].to(device)
    attention_mask = chunked["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    pooled_chunks = mean_pool_last_hidden(outputs.hidden_states[-1], attention_mask)

    aggregated_logits = logits.mean(dim=0, keepdim=True)
    aggregated_probs = probs.mean(dim=0, keepdim=True)
    aggregated_embedding = pooled_chunks.mean(dim=0, keepdim=True)

    pred_class = int(torch.argmax(aggregated_probs, dim=1).cpu().item())

    return {
        "pred_label_id": pred_class,
        "prob_ham": float(aggregated_probs[0, 0].cpu().item()),
        "prob_spam": float(aggregated_probs[0, 1].cpu().item()),
        "num_chunks": int(chunked["num_chunks"]),
        "tokenized_length": int(chunked["tokenized_length"]),
        "used_sliding_window": int(chunked["used_sliding_window"]),
        "embedding": aggregated_embedding.cpu().numpy()[0],
    }


def run_model_on_split(
    df: pd.DataFrame,
    model_name: str,
    model_dir: str,
    tokenizer_dir: str,
) -> pd.DataFrame:
    check_file_exists(model_dir)
    check_file_exists(tokenizer_dir)

    print(f"Loading {model_name} model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        output_hidden_states=True,
    )
    model.to(DEVICE)
    model.eval()

    dataset = StructuredTextDataset(df)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_samples,
    )

    rows = []
    total_batches = len(dataloader)

    for batch_idx, batch_samples in enumerate(dataloader, start=1):
        if batch_idx % 40 == 0 or batch_idx == total_batches:
            print(f"{model_name}: batch {batch_idx}/{total_batches}")

        for sample in batch_samples:
            result = forward_one_sample(
                model=model,
                tokenizer=tokenizer,
                text=sample["text"],
                max_length=BASE_MAX_LENGTH,
                stride=SLIDING_WINDOW_STRIDE,
                device=DEVICE,
            )

            row = {
                "lead_id": int(sample["lead_id"]),
                "message_id": int(sample["message_id"]),
                f"{model_name}_pred_label_id": result["pred_label_id"],
                f"{model_name}_prob_ham": result["prob_ham"],
                f"{model_name}_prob_spam": result["prob_spam"],
                f"{model_name}_num_chunks": result["num_chunks"],
                f"{model_name}_tokenized_length": result["tokenized_length"],
                f"{model_name}_used_sliding_window": result["used_sliding_window"],
            }

            emb = result["embedding"]
            for i, value in enumerate(emb):
                row[f"{model_name}_emb_{i}"] = float(value)

            rows.append(row)

    feature_df = pd.DataFrame(rows)

    if len(feature_df) != len(df):
        raise ValueError(
            f"{model_name}: output row count mismatch. "
            f"Expected {len(df)}, got {len(feature_df)}"
        )

    return feature_df.copy()


# BUILD FEATURE SPLIT

def build_feature_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    print(f"\nProcessing {split_name} split...")

    bert_df = run_model_on_split(
        df=df,
        model_name="bert",
        model_dir=BERT_MODEL_DIR,
        tokenizer_dir=BERT_TOKENIZER_DIR,
    )

    roberta_df = run_model_on_split(
        df=df,
        model_name="roberta",
        model_dir=ROBERTA_MODEL_DIR,
        tokenizer_dir=ROBERTA_TOKENIZER_DIR,
    )

    merged = df[["lead_id", "message_id"]].copy()
    merged = merged.merge(bert_df, on=["lead_id", "message_id"], how="inner")
    merged = merged.merge(roberta_df, on=["lead_id", "message_id"], how="inner")

    if len(merged) != len(df):
        raise ValueError(
            f"{split_name}: merged row count mismatch. "
            f"Expected {len(df)}, got {len(merged)}"
        )

    return merged.copy()


# MAIN

def main() -> None:
    create_dir(NLP_FEATURES_DIR)

    print(f"Using device: {DEVICE}")

    train_df = load_structured_split(TRAIN_STRUCTURED_PATH, "train_structured.csv")
    val_df = load_structured_split(VAL_STRUCTURED_PATH, "val_structured.csv")
    test_df = load_structured_split(TEST_STRUCTURED_PATH, "test_structured.csv")

    print(f"Train structured shape: {train_df.shape}")
    print(f"Validation structured shape: {val_df.shape}")
    print(f"Test structured shape: {test_df.shape}")

    train_features = build_feature_split(train_df, "train")
    val_features = build_feature_split(val_df, "val")
    test_features = build_feature_split(test_df, "test")

    train_features.to_csv(TRAIN_OUT, index=False)
    val_features.to_csv(VAL_OUT, index=False)
    test_features.to_csv(TEST_OUT, index=False)

    summary = {
        "device": str(DEVICE),
        "base_max_length": BASE_MAX_LENGTH,
        "sliding_window_stride": SLIDING_WINDOW_STRIDE,
        "train_shape": list(train_features.shape),
        "val_shape": list(val_features.shape),
        "test_shape": list(test_features.shape),
        "files_created": [
            TRAIN_OUT,
            VAL_OUT,
            TEST_OUT,
        ],
    }

    save_json(SUMMARY_OUT, summary)

    print("\nDone.")
    print(TRAIN_OUT)
    print(VAL_OUT)
    print(TEST_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()