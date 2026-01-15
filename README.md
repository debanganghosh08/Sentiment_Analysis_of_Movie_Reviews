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



---

## 🔄 Project Workflow


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

## ✅ Final Note (Engineer-to-Engineer)

What you now have is **two solid versions of the same project**:

- **Colab version** → submission-friendly, simple, fast
- **Local version** → research-grade, modular, professional

That’s exactly how **real-world ML projects** are done.

If you want next, I can:
- polish this README for recruiters
- create a **project architecture section for portfolio**
- help you explain **why classic NLP worked so well**

Just tell me.
