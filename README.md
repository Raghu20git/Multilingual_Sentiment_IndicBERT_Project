# Multilingual Robustness Analysis of Sentiment Models in Indian Ride-Hailing Reviews & Robustness Study of IndicBERT for Code-Mixed Indian User Text

## Overview

This project investigates the robustness of sentiment classification systems in the context of Indian ride-hailing reviews, with particular emphasis on multilingual and code-mixed text — a common characteristic of real-world user-generated content in India.

Two complementary modeling paradigms are studied:

- **Classical baseline:** TF-IDF + Logistic Regression  
- **Transformer model:** IndicBERT fine-tuning  

The primary objective is to examine how sentiment models trained largely on English data behave when exposed to linguistically diverse and noisy inputs.


## Motivation

Most sentiment analysis systems are developed and evaluated on clean English datasets. However, Indian user reviews frequently exhibit:

- Multilingual usage  
- Code-mixing (e.g., Hinglish)  
- Informal spelling and noise  
- Transliteration of Indic languages  

These characteristics often lead to performance degradation in standard NLP pipelines.

This project aims to systematically probe these weaknesses in the context of ride-hailing platforms.


## Datasets

### 1️. Primary Dataset

**Indian Ride Hailing Driver Reviews**  
https://www.kaggle.com/datasets/abubakkar01/indian-ride-hailing-driver-reviews

**Usage:**

- Main supervised training  
- Real-world noisy review distribution  

**Note:** The training data is predominantly English. Multilingual evaluation is therefore used as a robustness probe rather than full multilingual training.

### 2️. Code-Mixed Evaluation Set (Included)

A small synthetic Hinglish-style dataset is provided in: code_mixed_test.csv

**Purpose:**

- Stress-test model robustness  
- Evaluate behavior on mixed-language inputs  
- Identify failure patterns 


## Methodology  

### Baseline Model

- Unicode-aware preprocessing  
- Word + character TF-IDF  
- Logistic Regression classifier
- Error analysis pipeline

Character n-grams are included to improve robustness to spelling variation and transliteration.

### Advanced Model

- Pretrained **IndicBERT**
- Fine-tuned for 3-class sentiment  
- Early stopping with validation monitoring  
- Multilingual stress evaluation  
- Confusion matrix analysis  

This enables comparison between classical sparse methods and pretrained multilingual transformers.


## Evaluation

The following metrics are reported:

- Accuracy  
- Weighted F1-score  
- Confusion matrix  
- Qualitative error analysis  
- **Quantitative code-mixed accuracy**  

The code-mixed evaluation provides a direct measure of cross-lingual robustness.


## Results

### TF-IDF Baseline

**Accuracy & Classification Report (multilingual_sentiment):**

![TF-IDF Results](<Images/Accuracy and Classification Report (multilingual_sentiment).png>)

**Confusion Matrix (multilingual_sentiment):**

![TF-IDF CM](<Images/Confusion Matrix (indicbert_sentiment).png>)

**Error Analysis & Multilingual Stress Test (multilingual_sentiment):**

![TF-IDF Errors](<Images/Sample Errors and MST (multilingual_sentiment).png>)

### IndicBERT Model

**Confusion Matrix (indicbert_sentiment):**

![IndicBERT CM](<Images/Confusion Matrix (indicbert_sentiment).png>)

**Final Metrics & Multilingual Test (indicbert_sentiment):**

![IndicBERT Metrics](<Images/Final Metrics and Multilingual Test (indicbert_sentiment).png>)

**Code-Mixed Quantitative Evaluation**

![IndicBERT Code-Mixed Quantitative Evaluation](<Images/Code-Mixed Quantitative Evaluation (indicbert_sentiment).png>)


## Key Observations

- Strong performance is observed on in-domain English reviews.  
- Performance degrades on code-mixed and transliterated inputs.  
- Character n-grams improve classical model robustness.  
- IndicBERT shows better multilingual handling but still struggles with noisy Hinglish text.  
- Quantitative evaluation confirms measurable performance drop on code-mixed inputs

These findings highlight the gap between benchmark NLP performance and real Indian user data. 


## How to Run

### Step 1 - Install dependencies by running "pip install -r requirements.txt" in the command prompt

### Step 2 - Prepare dataset by downloading the datasets from "https://www.kaggle.com/datasets/abubakkar01/indian-ride-hailing-driver-reviews" and by running "python prepare_dataset.py" in the command prompt or run prepare_dataset.py directly after downloading the dataset from kaggle

### Step 3 - Run baseline model by running "python multilingual_sentiment.py" in the command prompt or run multilingual_sentiment.py directly

### Step 4 - Run IndicBERT model by running "python indicbert_sentiment.py" in the command prompt or run indicbert_sentiment.py directly 


## Project Structure

Multilingual_Sentiment_Project/
│
├── .vscode/
│   └── settings.json
├── Dataset/
│   ├── code_mixed_test.csv
│   └── indian_ride_hailing_services_analysis.csv
├── Images
├── .gitignore
├── prepare_dataset.py
├── multilingual_sentiment.py
├── indicbert_sentiment.py
├── requirements.txt
├── README.md
└── Report.md
