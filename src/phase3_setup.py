import os

def create_dir(path: str):
    os.makedirs(path, exist_ok=True)


def setup_phase3_directories(project_root: str):
    outputs_dir = os.path.join(project_root, "outputs")
    phase3_dir = os.path.join(outputs_dir, "phase3")

    # BERT structure
    bert_base = os.path.join(phase3_dir, "bert")
    create_dir(os.path.join(bert_base, "model"))
    create_dir(os.path.join(bert_base, "tokenizer"))
    create_dir(os.path.join(bert_base, "metrics"))
    create_dir(os.path.join(bert_base, "predictions"))
    create_dir(os.path.join(bert_base, "embeddings"))
    create_dir(os.path.join(bert_base, "config"))

    # RoBERTa structure
    roberta_base = os.path.join(phase3_dir, "roberta")
    create_dir(os.path.join(roberta_base, "model"))
    create_dir(os.path.join(roberta_base, "tokenizer"))
    create_dir(os.path.join(roberta_base, "metrics"))
    create_dir(os.path.join(roberta_base, "predictions"))
    create_dir(os.path.join(roberta_base, "embeddings"))
    create_dir(os.path.join(roberta_base, "config"))

    # NLP features
    create_dir(os.path.join(phase3_dir, "nlp_features"))

    # XGBoost structure
    xgb_base = os.path.join(phase3_dir, "xgboost")
    create_dir(os.path.join(xgb_base, "model"))
    create_dir(os.path.join(xgb_base, "metrics"))
    create_dir(os.path.join(xgb_base, "shap"))
    create_dir(os.path.join(xgb_base, "predictions"))

    print("\nPhase 3 directory structure created successfully.\n")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_phase3_directories(PROJECT_ROOT)