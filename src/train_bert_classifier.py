import os
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# CONFIG

@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    random_state: int = 42

    batch_size: int = 16
    eval_batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0

    base_max_length: int = 256
    sliding_window_stride: int = 192

    log_every_batches: int = 40
    num_labels: int = 2
    label_map: Dict[str, int] = None
    id2label: Dict[int, str] = None

    use_fp16: bool = True

    def __post_init__(self):
        if self.label_map is None:
            self.label_map = {"ham": 0, "spam": 1}
        if self.id2label is None:
            self.id2label = {0: "ham", 1: "spam"}


CFG = Config()


# PATHS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PHASE2_DIR = os.path.join(OUTPUTS_DIR, "phase2")
PHASE3_DIR = os.path.join(OUTPUTS_DIR, "phase3")

TRAIN_NLP_PATH = os.path.join(PHASE2_DIR, "train_nlp.csv")
VAL_NLP_PATH = os.path.join(PHASE2_DIR, "val_nlp.csv")
TEST_NLP_PATH = os.path.join(PHASE2_DIR, "test_nlp.csv")

BERT_DIR = os.path.join(PHASE3_DIR, "bert")
BERT_MODEL_DIR = os.path.join(BERT_DIR, "model")
BERT_TOKENIZER_DIR = os.path.join(BERT_DIR, "tokenizer")
BERT_METRICS_DIR = os.path.join(BERT_DIR, "metrics")
BERT_PREDICTIONS_DIR = os.path.join(BERT_DIR, "predictions")
BERT_EMBEDDINGS_DIR = os.path.join(BERT_DIR, "embeddings")
BERT_CONFIG_DIR = os.path.join(BERT_DIR, "config")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SETUP

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_phase3_directories() -> None:
    create_dir(PHASE3_DIR)

    # bert/
    create_dir(BERT_MODEL_DIR)
    create_dir(BERT_TOKENIZER_DIR)
    create_dir(BERT_METRICS_DIR)
    create_dir(BERT_PREDICTIONS_DIR)
    create_dir(BERT_EMBEDDINGS_DIR)
    create_dir(BERT_CONFIG_DIR)

    # roberta/
    roberta_base = os.path.join(PHASE3_DIR, "roberta")
    create_dir(os.path.join(roberta_base, "model"))
    create_dir(os.path.join(roberta_base, "tokenizer"))
    create_dir(os.path.join(roberta_base, "metrics"))
    create_dir(os.path.join(roberta_base, "predictions"))
    create_dir(os.path.join(roberta_base, "embeddings"))
    create_dir(os.path.join(roberta_base, "config"))

    # nlp_features/
    create_dir(os.path.join(PHASE3_DIR, "nlp_features"))

    # xgboost/
    xgb_base = os.path.join(PHASE3_DIR, "xgboost")
    create_dir(os.path.join(xgb_base, "model"))
    create_dir(os.path.join(xgb_base, "metrics"))
    create_dir(os.path.join(xgb_base, "shap"))
    create_dir(os.path.join(xgb_base, "predictions"))


def format_elapsed(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def check_required_columns(df: pd.DataFrame, required_cols: List[str], name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


# DATA

def load_split(path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = [
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
    check_required_columns(df, required_cols, split_name)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce").astype(int)
    df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce").astype(int)
    return df.reset_index(drop=True)


class NLPSampleDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "message_id": int(row["message_id"]),
            "source_dataset": str(row["source_dataset"]),
            "text": str(row["text"]),
            "label": str(row["label"]),
            "label_id": int(row["label_id"]),
            "token_count": int(row["token_count"]),
            "needs_sliding_window": int(row["needs_sliding_window"]),
            "too_long_for_single_512_pass": int(row["too_long_for_single_512_pass"]),
        }


def collate_samples(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch

# TOKENIZATION HELPERS
def add_special_tokens_manually(tokenizer, token_ids: List[int]) -> List[int]:
    """
    Manually wrap a single sequence with model special tokens.
    For BERT single sequence: [CLS] tokens [SEP]
    """
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Tokenizer must have cls_token_id and sep_token_id.")

    return [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]

# TOKENIZATION WITH SELECTIVE SLIDING WINDOW

def build_chunk_inputs(
    text: str,
    tokenizer,
    max_length: int,
    stride: int,
    needs_sliding_window_flag: int,
) -> Dict[str, torch.Tensor]:
    """
    Selective sliding window:
    - if token length <= max_length, use one chunk
    - if token length > max_length, use sliding windows
    """

    # Tokenize without truncation and suppress tokenizer max-length warning
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]

    token_len = len(token_ids)

    # For BERT single sequence: [CLS] + tokens + [SEP]
    special_tokens_count = 2
    content_max_len = max_length - special_tokens_count

    if content_max_len <= 0:
        raise ValueError("Invalid max_length. Must be greater than number of special tokens.")

    use_sliding = (needs_sliding_window_flag == 1) or (token_len > content_max_len)

    # ---------------------------------
    # Single-pass path
    # ---------------------------------
    if not use_sliding:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            verbose=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "num_chunks": 1,
            "tokenized_length": token_len,
            "used_sliding_window": 0,
        }

    # ---------------------------------
    # Sliding-window path
    # ---------------------------------
    if stride >= content_max_len:
        raise ValueError(
            f"sliding_window_stride ({stride}) must be smaller than content window size ({content_max_len})."
        )

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have pad_token_id for padded chunk batching.")

    chunk_input_ids = []
    chunk_attention_masks = []

    start = 0
    while start < token_len:
        end = min(start + content_max_len, token_len)
        chunk_token_ids = token_ids[start:end]

        # Add [CLS] and [SEP] manually
        full_ids = add_special_tokens_manually(tokenizer, chunk_token_ids)

        attention_mask = [1] * len(full_ids)

        # Pad to max_length
        pad_len = max_length - len(full_ids)
        if pad_len < 0:
            # Safety fallback, though this should not happen
            full_ids = full_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        else:
            full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
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


# MODEL HELPERS

def mean_pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def forward_one_sample(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    max_length: int,
    stride: int,
    device: torch.device,
    compute_loss: bool = True,
) -> Dict[str, Any]:
    chunked = build_chunk_inputs(
        text=sample["text"],
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        needs_sliding_window_flag=sample["needs_sliding_window"],
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

    result: Dict[str, Any] = {
        "aggregated_logits": aggregated_logits,
        "aggregated_probs": aggregated_probs,
        "aggregated_embedding": aggregated_embedding,
        "num_chunks": chunked["num_chunks"],
        "tokenized_length": chunked["tokenized_length"],
        "used_sliding_window": chunked["used_sliding_window"],
    }

    if compute_loss:
        labels = torch.tensor([sample["label_id"]], dtype=torch.long, device=device)
        loss = F.cross_entropy(aggregated_logits, labels)
        result["loss"] = loss

    return result


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# TRAIN / EVAL

def run_train_epoch(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch_idx: int,
    cfg: Config,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.train()

    epoch_start = time.time()
    batch_losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    epoch_rows: List[Dict[str, Any]] = []

    total_batches = len(dataloader)

    print(f"\n======== Epoch {epoch_idx + 1} / {cfg.epochs} ========")
    print("Training...")

    for batch_idx, batch_samples in enumerate(dataloader, start=1):
        optimizer.zero_grad()

        sample_losses: List[torch.Tensor] = []

        for sample in batch_samples:
            if cfg.use_fp16 and DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    result = forward_one_sample(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        max_length=cfg.base_max_length,
                        stride=cfg.sliding_window_stride,
                        device=DEVICE,
                        compute_loss=True,
                    )
            else:
                result = forward_one_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    max_length=cfg.base_max_length,
                    stride=cfg.sliding_window_stride,
                    device=DEVICE,
                    compute_loss=True,
                )

            sample_losses.append(result["loss"])

            prob_spam = float(result["aggregated_probs"][0, 1].detach().cpu().item())
            pred_class = int(torch.argmax(result["aggregated_probs"], dim=1).detach().cpu().item())

            y_true.append(int(sample["label_id"]))
            y_pred.append(pred_class)

            epoch_rows.append({
                "message_id": int(sample["message_id"]),
                "source_dataset": sample["source_dataset"],
                "label": sample["label"],
                "label_id": int(sample["label_id"]),
                "pred_label_id": pred_class,
                "pred_label": CFG.id2label[pred_class],
                "prob_ham": float(result["aggregated_probs"][0, 0].detach().cpu().item()),
                "prob_spam": prob_spam,
                "num_chunks": int(result["num_chunks"]),
                "tokenized_length": int(result["tokenized_length"]),
                "used_sliding_window": int(result["used_sliding_window"]),
            })

        batch_loss = torch.stack(sample_losses).mean()

        if cfg.use_fp16 and DEVICE.type == "cuda":
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        scheduler.step()

        batch_losses.append(float(batch_loss.detach().cpu().item()))

        if batch_idx % cfg.log_every_batches == 0 or batch_idx == total_batches:
            elapsed = format_elapsed(time.time() - epoch_start)
            print(f"  Batch {batch_idx:>5,}  of  {total_batches:>5,}.    Elapsed: {elapsed}.")

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(batch_losses)) if batch_losses else 0.0
    metrics["epoch_time_seconds"] = time.time() - epoch_start

    print()
    print(f"  Average training loss: {metrics['loss']:.4f}")
    print(f"  Training Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Training F1-Score: {metrics['f1']:.4f}")
    print(f"  Training epoch took: {format_elapsed(metrics['epoch_time_seconds'])}")

    return metrics, epoch_rows


@torch.no_grad()
def run_eval(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    split_name: str,
    cfg: Config,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    model.eval()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[Dict[str, Any]] = []
    embedding_rows: List[np.ndarray] = []

    print(f"\nRunning {split_name.capitalize()}...")

    for batch_samples in dataloader:
        for sample in batch_samples:
            result = forward_one_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                max_length=cfg.base_max_length,
                stride=cfg.sliding_window_stride,
                device=DEVICE,
                compute_loss=True,
            )

            loss_value = float(result["loss"].detach().cpu().item())
            losses.append(loss_value)

            probs = result["aggregated_probs"].detach().cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            embedding = result["aggregated_embedding"].detach().cpu().numpy()[0]

            y_true.append(int(sample["label_id"]))
            y_pred.append(pred_class)
            embedding_rows.append(embedding)

            rows.append({
                "message_id": int(sample["message_id"]),
                "source_dataset": sample["source_dataset"],
                "text": sample["text"],
                "label": sample["label"],
                "label_id": int(sample["label_id"]),
                "pred_label_id": pred_class,
                "pred_label": CFG.id2label[pred_class],
                "prob_ham": float(probs[0]),
                "prob_spam": float(probs[1]),
                "num_chunks": int(result["num_chunks"]),
                "tokenized_length": int(result["tokenized_length"]),
                "used_sliding_window": int(result["used_sliding_window"]),
                "loss": loss_value,
            })

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0

    print(f"  {split_name.capitalize()} Loss: {metrics['loss']:.4f}")
    print(f"  {split_name.capitalize()} Accuracy: {metrics['accuracy']:.4f}")
    print(f"  {split_name.capitalize()} F1-Score: {metrics['f1']:.4f}")
    print(f"  {split_name.capitalize()} Precision: {metrics['precision']:.4f}")
    print(f"  {split_name.capitalize()} Recall: {metrics['recall']:.4f}")

    pred_df = pd.DataFrame(rows)
    embeddings = np.vstack(embedding_rows) if embedding_rows else np.empty((0, 0), dtype=np.float32)

    return metrics, pred_df, embeddings


# SAVING

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_epoch_history(history: List[Dict[str, Any]]) -> None:
    pd.DataFrame(history).to_csv(
        os.path.join(BERT_METRICS_DIR, "training_history.csv"),
        index=False
    )
    save_json(
        os.path.join(BERT_METRICS_DIR, "training_history.json"),
        {"history": history}
    )


def save_split_outputs(
    split_name: str,
    pred_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    pred_path = os.path.join(BERT_PREDICTIONS_DIR, f"{split_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    embedding_matrix_path = os.path.join(BERT_EMBEDDINGS_DIR, f"{split_name}_embeddings.npy")
    np.save(embedding_matrix_path, embeddings)

    embedding_meta = pred_df[[
        "message_id",
        "source_dataset",
        "label",
        "label_id",
        "pred_label",
        "pred_label_id",
        "prob_ham",
        "prob_spam",
        "num_chunks",
        "tokenized_length",
        "used_sliding_window",
    ]].copy()
    embedding_meta.to_csv(
        os.path.join(BERT_EMBEDDINGS_DIR, f"{split_name}_embedding_metadata.csv"),
        index=False
    )


def save_metrics_file(
    split_name: str,
    metrics: Dict[str, Any],
    pred_df: pd.DataFrame,
) -> None:
    cm = confusion_matrix(pred_df["label_id"], pred_df["pred_label_id"]).tolist()
    report = classification_report(
        pred_df["label_id"],
        pred_df["pred_label_id"],
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    payload = {
        "split": split_name,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    save_json(os.path.join(BERT_METRICS_DIR, f"{split_name}_metrics.json"), payload)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(BERT_METRICS_DIR, f"{split_name}_metrics.csv"),
        index=False
    )


def save_final_artifacts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    cfg: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    #model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_TOKENIZER_DIR)

    config_payload = asdict(cfg)
    config_payload["device"] = str(DEVICE)
    config_payload["train_rows"] = int(len(train_df))
    config_payload["val_rows"] = int(len(val_df))
    config_payload["test_rows"] = int(len(test_df))

    save_json(os.path.join(BERT_CONFIG_DIR, "train_config.json"), config_payload)
    save_json(os.path.join(BERT_CONFIG_DIR, "label_map.json"), {
        "label_map": cfg.label_map,
        "id2label": {str(k): v for k, v in cfg.id2label.items()},
    })


# MAIN

def main() -> None:
    setup_phase3_directories()
    set_seed(CFG.random_state)

    print(f"Using device: {DEVICE}")

    train_df = load_split(TRAIN_NLP_PATH, "train_nlp.csv")
    val_df = load_split(VAL_NLP_PATH, "val_nlp.csv")
    test_df = load_split(TEST_NLP_PATH, "test_nlp.csv")

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_labels,
        output_hidden_states=True,
    )
    model.to(DEVICE)

    train_dataset = NLPSampleDataset(train_df)
    val_dataset = NLPSampleDataset(val_df)
    test_dataset = NLPSampleDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.eval_batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.eval_batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        eps=CFG.adam_epsilon,
        weight_decay=CFG.weight_decay,
    )

    total_training_steps = len(train_loader) * CFG.epochs
    warmup_steps = int(total_training_steps * CFG.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(CFG.use_fp16 and DEVICE.type == "cuda"))

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1

    overall_start = time.time()

    for epoch_idx in range(CFG.epochs):
        train_metrics, _ = run_train_epoch(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch_idx=epoch_idx,
            cfg=CFG,
        )

        val_metrics, val_pred_df, val_embeddings = run_eval(
            model=model,
            tokenizer=tokenizer,
            dataloader=val_loader,
            split_name="validation",
            cfg=CFG,
        )

        epoch_record = {
            "epoch": epoch_idx + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(epoch_record)
        save_epoch_history(history)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch_idx + 1

            model.save_pretrained(BERT_MODEL_DIR)
            tokenizer.save_pretrained(BERT_TOKENIZER_DIR)

            save_split_outputs("validation_best", val_pred_df, val_embeddings)
            save_metrics_file("validation_best", val_metrics, val_pred_df)

            save_json(
                os.path.join(BERT_CONFIG_DIR, "best_checkpoint_info.json"),
                {
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                }
            )

    print("\nTraining complete.")
    print(f"Total training took: {format_elapsed(time.time() - overall_start)}")
    print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

    print("\nReloading best saved BERT model for final train/validation/test inference...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_DIR,
        num_labels=CFG.num_labels,
        output_hidden_states=True,
    )
    best_model.to(DEVICE)

    train_metrics, train_pred_df, train_embeddings = run_eval(
        model=best_model,
        tokenizer=tokenizer,
        dataloader=train_loader,
        split_name="train",
        cfg=CFG,
    )
    val_metrics, val_pred_df, val_embeddings = run_eval(
        model=best_model,
        tokenizer=tokenizer,
        dataloader=val_loader,
        split_name="validation",
        cfg=CFG,
    )
    test_metrics, test_pred_df, test_embeddings = run_eval(
        model=best_model,
        tokenizer=tokenizer,
        dataloader=test_loader,
        split_name="test",
        cfg=CFG,
    )

    save_split_outputs("train", train_pred_df, train_embeddings)
    save_split_outputs("validation", val_pred_df, val_embeddings)
    save_split_outputs("test", test_pred_df, test_embeddings)

    save_metrics_file("train", train_metrics, train_pred_df)
    save_metrics_file("validation", val_metrics, val_pred_df)
    save_metrics_file("test", test_metrics, test_pred_df)

    save_final_artifacts(
        #model=best_model,
        tokenizer=tokenizer,
        cfg=CFG,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    save_json(
        os.path.join(BERT_CONFIG_DIR, "run_summary.json"),
        {
            "model_name": CFG.model_name,
            "device": str(DEVICE),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "artifacts_saved": {
                "model_dir": BERT_MODEL_DIR,
                "tokenizer_dir": BERT_TOKENIZER_DIR,
                "metrics_dir": BERT_METRICS_DIR,
                "predictions_dir": BERT_PREDICTIONS_DIR,
                "embeddings_dir": BERT_EMBEDDINGS_DIR,
                "config_dir": BERT_CONFIG_DIR,
            },
        }
    )

    print("\nSaved BERT artifacts:")
    print(BERT_MODEL_DIR)
    print(BERT_TOKENIZER_DIR)
    print(BERT_METRICS_DIR)
    print(BERT_PREDICTIONS_DIR)
    print(BERT_EMBEDDINGS_DIR)
    print(BERT_CONFIG_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()