# Sentiment Analysis of Movie Reviews

This repository contains a complete, research-oriented NLP project for
**sentiment analysis of IMDb movie reviews**, implemented using
classical machine learning models and modern NLP practices.

The project follows a **modular local architecture**, separating
data handling, preprocessing, modeling, evaluation, and reporting.

---

## 📌 Project Objective

To automatically classify movie reviews as **Positive** or **Negative**
based on their textual content, and to compare different NLP approaches
in a structured and reproducible manner.

---

## 📂 Project Structure
<img width="2182" height="2934" alt="image" src="https://github.com/user-attachments/assets/16b8116c-b69f-4db6-b420-fb4f01ccf24f" />
---
<img width="2232" height="2124" alt="image" src="https://github.com/user-attachments/assets/a1aafe77-07d6-4f74-8632-7c1f78b3caaa" />

---

## 🔄 Project Workflow
<img width="1998" height="1747" alt="image" src="https://github.com/user-attachments/assets/c355ee3e-cba2-4009-85fb-1ad6b8f3a4a4" />

---

## 🧠 Models Used

- **Logistic Regression (TF-IDF)**
- **Support Vector Machine (TF-IDF)**
- **DistilBERT (Transformer, optional advanced model)**

---

## 📊 Results Summary

| Model | Test Accuracy |
|------|---------------|
| Logistic Regression | ~0.90 |
| SVM | ~0.90 |
| DistilBERT | ~0.91 |

Classical TF-IDF based models perform competitively while being faster
and more interpretable.

---

## 🚀 How to Run Locally

1. Create environment:
   ```bash
   conda env create -f environment.yml
   conda activate movie_sentiment_env
Run notebooks in order:

01_data_exploration.ipynb

02_text_preprocessing.ipynb

03_modeling_baseline.ipynb

04_error_analysis.ipynb

06_model_comparison_and_insights.ipynb

---
