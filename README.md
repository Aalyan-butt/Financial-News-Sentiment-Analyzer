# 💰 Financial News Sentiment Analyzer

<p align="center">
  <b>📈 NLP | 📰 Financial News | 💬 Sentiment Analysis | 🐍 Python</b><br><br>
  Analyze financial news headlines and classify their sentiment as <strong>Positive</strong>, <strong>Negative</strong>, or <strong>Neutral</strong>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  <img src="https://img.shields.io/badge/NLP-Scikit--learn-yellow?logo=scikit-learn">
  <img src="https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit">
</p>

---

## 🚀 Project Overview

The **Financial News Sentiment Analyzer** uses **natural language processing (NLP)** techniques to determine the sentiment of financial news headlines. This helps traders, investors, and analysts gauge market mood from textual data.

The system classifies each headline into one of three categories:

* ✅ **Positive**
* ⚠️ **Neutral**
* ❌ **Negative**

---

## 🧠 Features

✅ Clean and preprocess financial news text
✅ Apply feature extraction (TF-IDF or word embeddings)
✅ Train classifiers (Logistic Regression, Naïve Bayes, or SVM)
✅ Multiclass sentiment prediction
✅ Model evaluation using accuracy, F1-score, and confusion matrix
✅ (Optional) Live prediction interface with Streamlit

---

## 📁 Project Structure

```
financial-news-sentiment-analyzer/
│
├── data/                   # Raw and preprocessed datasets
├── notebooks/              # Jupyter notebooks for EDA & modeling
├── models/                 # Saved machine learning models
├── app/                    # Streamlit or Flask app
├── main.py                 # Script for training and evaluating model
├── README.md               # Project documentation

```

---

## 📰 Dataset

📚 **Financial PhraseBank** (or similar curated dataset)

* Labeled financial news headlines
* Sentiment categories: Positive, Negative, Neutral
* Suitable for real-world financial sentiment tasks

🔗 [Example Dataset on Kaggle](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)

---

## ⚙️ Tech Stack

| Tool              | Purpose                                     |
| ----------------- | ------------------------------------------- |
| 🐍 Python         | Main programming language                   |
| 🧪 Scikit-learn   | ML algorithms and preprocessing             |
| 🭹 NLTK / spaCy   | NLP preprocessing (tokenization, stopwords) |
| 📊 Pandas / NumPy | Data manipulation and analysis              |
| 🌐 Streamlit      | Optional web app interface                  |

---

## 📈 Model Evaluation

Evaluation metrics include:

* 🌟 **Accuracy**: Overall correctness
* ▰️ **F1-Score (Macro)**: Balanced performance across all sentiment classes
* 📊 **Confusion Matrix**: Visual insight into class-wise prediction quality

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Aalyan-butt/Financial-News-Sentiment-Analyzer
cd financial-news-sentiment-analyzer
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```



### 3. Train and Evaluate the Model

```bash
python main.py
```

### 4. (Optional) Launch Streamlit App

```bash
streamlit run app/app.py
```

---

## 💡 Example Usage

> **Input:** "Apple's quarterly revenue beats Wall Street expectations"
> **Predicted Sentiment:** ✅ Positive

> **Input:** "Inflation fears drag down global stock markets"
> **Predicted Sentiment:** ❌ Negative

---

## 🎯 Future Improvements

* [ ] Integrate deep learning (LSTM, BERT) for better context awareness
* [ ] Real-time financial news feed integration
* [ ] Sentiment-based stock prediction model
* [ ] Dashboard with interactive charts and visualizations

---



## 🙌 Acknowledgments

* Financial PhraseBank Dataset
* Scikit-learn, NLTK, spaCy
* Streamlit team for easy deployment tools
* Kaggle community for shared datasets

---

## 👤 Author

**Aalyan Riasat**
📧 [aalyanriasatali@gmail.com](mailto:your.email@example.com)
🔗  • [GitHub](https://github.com/Aalyan-butt)

---
