# Multilingual Robustness Analysis of Sentiment Models in Indian Ride-Hailing Reviews & Robustness Study of IndicBERT for Code-Mixed Indian User Text

## 1. Problem Statement

Sentiment analysis systems trained on clean English corpora often exhibit performance degradation when deployed on multilingual and code-mixed user-generated content. This issue is especially pronounced in the Indian context, where users frequently mix languages (e.g., Hinglish) and employ transliterated Indic text.

This work investigates the robustness of both classical machine learning methods and pretrained multilingual transformers for sentiment classification in Indian ride-hailing reviews.


## 2. Objectives

The primary objectives of this study are:

- To build a sentiment classification pipeline for ride-hailing reviews  
- To evaluate cross-lingual generalization under predominantly English supervision  
- To quantitatively analyze model behavior on code-mixed inputs  
- To compare a classical TF-IDF baseline with IndicBERT  

**Note:** Since the training data is largely English, multilingual evaluation is treated as a robustness probe rather than full multilingual training.


## 3. Dataset

The experiments utilize:

- **Indian Ride Hailing Reviews (~11k samples)** — primary supervised dataset  
- **Synthetic Code-Mixed Evaluation Set** — used for quantitative robustness testing  
- **Multilingual Stress Sentences** — Hindi, Telugu, and Hinglish examples  

The primary dataset is predominantly English, which is an important limitation when interpreting multilingual performance.


## 4. Methodology

### 4.1 Baseline Model

The classical pipeline consists of:

- Unicode-aware text preprocessing  
- Word-level TF-IDF features  
- Character n-gram TF-IDF features  
- Logistic Regression classifier  
- Post-hoc error analysis  

Character n-grams are included to improve robustness to spelling variation and transliteration noise.

### 4.2 Advanced Model

The neural approach employs:

- Pretrained **IndicBERT**  
- Fine-tuning for 3-class sentiment classification  
- Early stopping based on validation performance  
- Multilingual stress testing  
- **Quantitative code-mixed evaluation**  

This enables comparison between sparse lexical models and pretrained multilingual representations.


## 5. Results

### 5.1 TF-IDF Baseline

**Accuracy & Classification Report (multilingual_sentiment):**

![TF-IDF Results](Images/Accuracy%20and%20Classification%20Report%20(multilingual_sentiment).png)

**Confusion Matrix (multilingual_sentiment):**

![TF-IDF CM](Images/Confusion%20Matrix%20(indicbert_sentiment).png)

**Error Analysis & Multilingual Stress Test (multilingual_sentiment):**

![TF-IDF Errors](Images/Sample%20Errors%20and%20MST%20(multilingual_sentiment).png)

### 5.2 IndicBERT Model

**Confusion Matrix (indicbert_sentiment):**

![IndicBERT CM](Images/Confusion%20Matrix%20(indicbert_sentiment).png)

**Final Metrics & Multilingual Test (indicbert_sentiment):**

![IndicBERT Metrics](Images/Final%20Metrics%20and%20Multilingual%20Test%20(indicbert_sentiment).png)

**Code-Mixed Quantitative Evaluation**

![IndicBERT Code-Mixed Quantitative Evaluation](<Images/Code-Mixed Quantitative Evaluation (indicbert_sentiment).png>)

The model achieves strong performance on the in-domain test set but shows measurable degradation on the code-mixed evaluation set, indicating reduced robustness under mixed-language conditions.


## 6. Observations

Key findings include:

- Strong performance on in-domain English reviews  
- Noticeable degradation on code-mixed and transliterated inputs  
- Character n-grams improve robustness of the classical model  
- IndicBERT handles multilingual text better but still struggles with noisy Hinglish  

These results highlight the gap between benchmark NLP performance and real Indian user data.


## 7. Limitations

This study has several limitations:

- Training data is predominantly English  
- Limited true Indic-language supervision  
- Small-scale code-mixed evaluation set  
- No explicit code-mixed fine-tuning  

These constraints should be considered when interpreting cross-lingual performance.


## 8. Future Work

Potential improvements include:

- Training on large-scale Indic sentiment datasets  
- Explicit fine-tuning on code-mixed corpora  
- Cross-lingual representation alignment  
- Lightweight multilingual deployment models  


## 9. Conclusion

This study demonstrates that while IndicBERT improves multilingual robustness relative to classical TF-IDF methods, significant challenges remain for real-world Indian code-mixed text. Quantitative evaluation confirms that models trained predominantly on English data exhibit measurable degradation under mixed-language conditions, underscoring the importance of linguistically diverse supervision.