# =========================================
# TF-IDF Multilingual Sentiment Analysis
# =========================================

import os
import joblib
import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack

nltk.download("stopwords")
from nltk.corpus import stopwords

# =========================================
# PATHS
# =========================================
DATA_PATH = r"C:\Users\sragh\Documents\Multilingual_Sentiment_Project\Dataset\indian_ride_hailing_services_analysis.csv"
MODEL_FILE = "saved_tfidf_model.joblib"

TEXT_COL = "review"
RATING_COL = "rating"

# =========================================
# CLEANING (CODE-MIX FRIENDLY)
# =========================================
stop_words_en = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # keep multilingual unicode
    text = re.sub(r"[^\w\s\u0900-\u097F\u0C00-\u0C7F]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    # remove only English stopwords
    tokens = [t for t in tokens if t not in stop_words_en]

    return " ".join(tokens)

# =========================================
# LOAD OR TRAIN MODEL
# =========================================
if os.path.exists(MODEL_FILE):
    print("Loading saved TF-IDF model...")

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    word_vectorizer = saved["word_vec"]
    char_vectorizer = saved["char_vec"]

else:
    print("No saved model found. Training from scratch...")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print(df.head())

    # -----------------------------
    # CREATE LABELS
    # -----------------------------
    def rating_to_label(r):
        if r >= 4:
            return "positive"
        elif r <= 2:
            return "negative"
        else:
            return "neutral"

    df["label"] = df[RATING_COL].apply(rating_to_label)

    # -----------------------------
    # CLEAN TEXT
    # -----------------------------
    df["clean_text"] = df[TEXT_COL].apply(clean_text)

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # -----------------------------
    # TF-IDF (WORD + CHARACTER)
    # -----------------------------
    word_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=3000
    )

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)

    X_train_vec = hstack([X_train_word, X_train_char])
    X_test_vec = hstack([X_test_word, X_test_char])

    print("Feature shape:", X_train_vec.shape)

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # -----------------------------
    # EVALUATION
    # -----------------------------
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("tfidf_confusion_matrix.png")
    plt.show()

    # -----------------------------
    # ERROR ANALYSIS
    # -----------------------------
    errors = pd.DataFrame({
        "text": X_test,
        "actual": y_test,
        "predicted": y_pred
    })

    print("\nSample Errors:")
    print(errors[errors["actual"] != errors["predicted"]].head(10))

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    joblib.dump(
        {
            "model": model,
            "word_vec": word_vectorizer,
            "char_vec": char_vectorizer,
        },
        MODEL_FILE,
    )

    print("TF-IDF model saved.")

# =========================================
# PREDICTION FUNCTION
# =========================================
def predict_sentiment(text):
    cleaned = clean_text(text)
    w = word_vectorizer.transform([cleaned])
    c = char_vectorizer.transform([cleaned])
    vec = hstack([w, c])
    return model.predict(vec)[0]

# =========================================
# MULTILINGUAL STRESS TEST
# =========================================
print("\n=== Multilingual Stress Test ===")

tests = [
    "This driver is very good",
    "यह ड्राइवर बहुत खराब है",
    "ఈ డ్రైవర్ చాలా మంచివాడు",
    "bahut acha driver tha",
    "service bilkul bakwas hai"
]

for t in tests:
    print(f"{t} → {predict_sentiment(t)}")