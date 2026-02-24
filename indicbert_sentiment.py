# =========================================
# IndicBERT Multilingual Sentiment Analysis
# =========================================

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# DEVICE AUTO-DETECTION
# =========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =========================================
# PATHS
# =========================================
DATA_PATH = r"C:\Users\sragh\Documents\Multilingual_Sentiment_Project\Dataset\indian_ride_hailing_services_analysis.csv"
MODEL_NAME = "ai4bharat/indic-bert"
MODEL_DIR = "saved_indicbert"

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# =========================================
# LABEL CREATION
# =========================================
def rating_to_label(r):
    if r >= 4:
        return 2
    elif r <= 2:
        return 0
    else:
        return 1

df["label"] = df["rating"].apply(rating_to_label)
df = df[["review", "label"]].dropna()

print("\nLabel distribution:")
print(df["label"].value_counts(normalize=True))

# =========================================
# TRAIN TEST SPLIT
# =========================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# =========================================
# LOAD OR CREATE MODEL
# =========================================
if os.path.exists(MODEL_DIR):
    print("Loading saved IndicBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    SKIP_TRAIN = True
else:
    print("No saved model found. Training from scratch...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    ).to(device)
    SKIP_TRAIN = False

# =========================================
# TOKENIZATION
# =========================================
def tokenize(batch):
    return tokenizer(
        batch["review"],
        truncation=True,
        max_length=128,
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =========================================
# METRICS
# =========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# =========================================
# TRAINING CONFIG (YOUR TUNED VERSION)
# =========================================
training_args = TrainingArguments(
    output_dir="./results",

    # tuned learning
    learning_rate=1.5e-5,
    warmup_ratio=0.06,
    weight_decay=0.01,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=6,
    max_grad_norm=1.0,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",

    logging_steps=50,
    logging_dir="./logs",
    seed=42,

    fp16=(device.type == "cuda"),
    dataloader_pin_memory=(device.type == "cuda"),
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# =========================================
# TRAIN ONLY IF NEEDED
# =========================================
if not SKIP_TRAIN:
    print("Training model...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

# =========================================
# EVALUATION
# =========================================
print("Evaluating...")
metrics = trainer.evaluate()
predictions = trainer.predict(test_ds)

y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# =========================================
# CONFUSION MATRIX (AUTO SAFE)
# =========================================
labels_present = sorted(np.unique(y_true))

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

display_labels = [label_map[l] for l in labels_present]

cm = confusion_matrix(y_true, y_pred, labels=labels_present)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(cmap="Blues")
plt.title("IndicBERT Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("\nFinal Metrics:", metrics)

# =========================================
# SAFE PREDICT
# =========================================
def predict(text):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()

    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return mapping[pred]

# =========================================
# CODE-MIXED QUANTITATIVE EVALUATION
# =========================================

def evaluate_code_mixed(csv_path=r"C:\Users\sragh\Documents\Multilingual_Sentiment_Project\Dataset\code_mixed_test.csv"):
    print("\n=== Code-Mixed Quantitative Evaluation ===")

    cm_df = pd.read_csv(csv_path)
    cm_df["label"] = cm_df["rating"].apply(rating_to_label)

    y_true = []
    y_pred = []

    for _, row in cm_df.iterrows():
        pred_label = predict(row["review"])
        true_label = {0: "negative", 1: "neutral", 2: "positive"}[row["label"]]

        y_true.append(true_label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)

    print("Code-mixed accuracy:", round(acc, 4))
    return acc

# =========================================
# MULTILINGUAL TEST
# =========================================
print("\n=== Multilingual Test ===")

tests = [
    "This driver is very good",
    "यह ड्राइवर बहुत खराब है",
    "ఈ డ్రైవర్ చాలా మంచివాడు",
    "bahut acha driver tha",
    "service bilkul bakwas hai"
]

for t in tests:
    print(f"{t} → {predict(t)}")

# =========================================
# RUN CODE-MIXED EVALUATION
# =========================================

cm_acc = evaluate_code_mixed()