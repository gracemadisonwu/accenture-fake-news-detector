# 📰 Fake/Real News Classifier with Natural Language Processing — Accenture

---

## 👥 Team Members

| Name | GitHub Handle | Contribution |
|------|--------------|--------------|
| Alexandru Soroium | @alexandrusoroium | Feature engineering, SVM model design and evaluation, project documentation |
| Grace Madison Wu | @gracemadisonwu | Data collection, exploratory data analysis (EDA), dataset documentation |
| Aniekan Inyang | @aniekanai | Data preprocessing, text cleaning, lemmatization, Naive Bayes Development, data validation |
| Karina Hernandez | @khern2005 | Model selection, hyperparameter tuning, training classical ML baselines |
| Arif Manawer | @arifmanawer | LSTM model development, deep learning experiments, performance analysis |
| Sumaiya Chowdhury | @sumaiyachow3 | BERT model experimentation, Random forest development, evaluation, results interpretation |

---

## 🎯 Project Highlights

- Built a fake vs real news classifier using **multiple NLP approaches**, including **TF-IDF + linear SVM**, **LSTM**, and a fine-tuned **BERT** sequence classification model on news titles and article bodies.
- After removing data leakage from the original “subject” feature, the best TF-IDF + linear SVM model on title and text reached about **99.6%** accuracy with near-perfect precision, recall, and ROC-AUC on a held-out test set.
- A deep LSTM model on combined title and text achieved roughly **99.3%** test accuracy, correctly classifying **8,919** out of **8,980** articles and showing stable learning across epochs.
- Engineered interpretable features such as **punctuation density**, **text length**, **seasonality** (year, month, weekday), and **contraction usage**, and used these insights to diagnose and reduce data leakage from tokens like **"Reuters"** and **"said"**, making the models more robust for an enterprise context at **Accenture**.

---

## 🛠 Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/AlexandruSoroium/accenture-ai-studio-fall2025.git
cd accenture-ai-studio-fall2025
```

### 2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Key libraries include **pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, tensorflow/keras, transformers, datasets, dateparser**.

### 4. Download and place the dataset

This project uses the **Fake.csv** and **True.csv** files. Each contains article **title, text, subject, and date**.

1. Download from Kaggle  
   https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data
2. Place them in a `/data` folder:

```
data/Fake.csv
data/True.csv
```

3. Ensure the notebook references the local paths (not Google Drive).

### 5. Run the notebook(s) or scripts

- Open `notebooks/AccentureJle.ipynb` and run all cells in order.  
- GPU is recommended for BERT training.

---

## 🧠 Project Overview

This project was completed as part of the Break Through Tech AI @ Cornell Tech Fall AI Studio program in collaboration with Accenture. The goal was to design and evaluate machine learning models that automatically classify online news articles as fake or real, focusing on scalable, explainable, and production-oriented NLP solutions.

The project explored classical and deep learning approaches, reducing data leakage to improve trustworthy real-world deployment. Detecting fake news supports **media, finance, and public-sector clients**, enabling early-warning systems and prioritizing high-risk articles.

---

## 📊 Data Exploration

The combined dataset contains ~**35,763** articles (**59% true / 41% fake**). The team:

- parsed dates into usable `datetime` objects  
- created **year, month, and day-of-week** features  
- analyzed distributions and token trends  
- identified leakage from **"Reuters"**, **"said"**, and **subject column bias**

Key EDA insights:

- True articles cluster around **2016–2017**, reflecting dataset design bias.
- Fake news appears more frequently on weekends.
- Removing biased tokens improves generalization.

---

## 🧩 Model Development

### 1. Preprocessing and Cleaning
- Combined datasets, added labels, dropped **subject** due to leakage.
- Removed boilerplate text and stopwords, lemmatized tokens.
- Added features: length counts, punctuation, contractions.

### 2. Classical Frequency-Based Models
- Built BoW and TF-IDF features for **title, text, and combined text**.
- Trained **Naive Bayes, Logistic Regression, Random Forests, SVM** baselines.
- Primary model: **ColumnTransformer + LinearSVC** with stratified splits.

### 3. Embedding and Deep Models
- Averaged **Word2Vec** embeddings for SVM classifier.
- Built an **LSTM** on combined sequences with validation early stopping.
- Fine-tuned **BERT (bert-base-uncased)** with class-balanced loss.

---

## 📈 Results & Key Findings

- **TF-IDF + SVM:** ~**99.6%** accuracy, strong ROC-AUC  
- **LSTM:** ~**99.3%** accuracy, 61 misclassified articles total  
- **Random Forest:** strong but no major gain over linear models  
- **BERT:** competitive but higher deployment cost

Leakage mitigation confirmed accuracy improvements were meaningful rather than artifact-driven.

---

## 🚀 Next Steps

- Add broader, fresher data across languages and time.
- Evaluate robustness against adversarial or paraphrased text.
- Explore topic fairness and performance subgroup analysis.
- Package best model into a production API with monitoring.

---

## 📝 License
This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements
Thanks to **Break Through Tech AI, Accenture, Isabel Heard, Jenna Hunte, the BTT instructional team, and our mentors** for guidance and support.
