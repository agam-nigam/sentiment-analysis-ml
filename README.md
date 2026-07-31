# 🤖 Sentiment Analysis Using Machine Learning

> A modern Machine Learning web application that classifies text into **Positive**, **Negative**, or **Neutral** sentiments using **TF-IDF Vectorization** and a **Linear Support Vector Machine (Linear SVM)**. The project includes an interactive **Streamlit dashboard** for real-time sentiment prediction and dataset exploration.

---

## 📌 Overview

This project implements a complete **Natural Language Processing (NLP)** pipeline for sentiment analysis. It preprocesses raw text, converts it into numerical feature vectors using **TF-IDF**, trains multiple machine learning models, evaluates their performance, and deploys the best-performing model through a user-friendly Streamlit interface.

The application enables users to:

- 🔍 Analyze the sentiment of custom text
- 📊 Explore the processed dataset
- 📈 View text statistics
- ⚡ Receive instant sentiment predictions

---

## ✨ Features

- 🎯 Real-time sentiment prediction
- 🧹 Text preprocessing and cleaning
- 📖 Stopword removal using NLTK
- 🔤 TF-IDF feature extraction
- 🤖 Machine Learning based classification
- 📊 Interactive Streamlit dashboard
- 📂 Dataset preview
- 📈 Text statistics
- ⚡ Fast inference using a saved model
- 💾 Model persistence using Joblib

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP | NLTK |
| Data Processing | Pandas, NumPy |
| Visualization | Streamlit |
| Model Persistence | Joblib |

---

## 📂 Project Structure

```text
Sentiment-Analysis-Using-Machine-Learning/
│
├── app.py
├── analysis.ipynb
├── sentimentdataset.csv
├── cleaned_sentiment_dataset.csv
├── best_sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── assets/
    ├── banner.png
    └── preview.png
```

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Collection

- Load sentiment dataset
- Remove unnecessary columns
- Handle missing values

### 2️⃣ Text Preprocessing

- Convert text to lowercase
- Remove URLs
- Remove mentions
- Remove hashtags
- Remove punctuation
- Remove numbers
- Remove stopwords
- Remove extra spaces

### 3️⃣ Feature Engineering

TF-IDF Vectorizer

```
max_features = 15000
ngram_range = (1,3)
min_df = 2
max_df = 0.95
```

### 4️⃣ Models Evaluated

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine
- Random Forest Classifier

### 5️⃣ Model Selection

The best-performing model is saved as:

```
best_sentiment_model.pkl
```

along with

```
tfidf_vectorizer.pkl
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Sentiment-Analysis-Using-Machine-Learning.git
```

Go inside the project

```bash
cd Sentiment-Analysis-Using-Machine-Learning
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Workflow

```text
User Input
     │
     ▼
Text Cleaning
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Linear SVM Model
     │
     ▼
Sentiment Prediction
```

---

## 📈 Sample Prediction

| Input | Predicted Sentiment |
|--------|---------------------|
| I absolutely loved this movie! | 😊 Positive |
| The service was terrible. | 😞 Negative |
| It was okay, nothing special. | 😐 Neutral |

---

## 📸 Application Preview

### Home Page

> Add screenshot here

```
assets/home.png
```

### Prediction

> Add screenshot here

```
assets/prediction.png
```

### Dataset Explorer

> Add screenshot here

```
assets/dataset.png
```

---

## 📋 Requirements

- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Joblib

Install using

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

- 🌐 Deploy on Streamlit Cloud
- 🤖 Deep Learning (LSTM/BERT)
- ☁️ REST API using FastAPI
- 📈 Advanced visualizations
- 📱 Mobile responsive dashboard
- 🌍 Multi-language sentiment analysis

---

## 👨‍💻 Author

**Agam Nigam**

Computer Engineering Undergraduate

Interested in:

- Machine Learning
- Artificial Intelligence
- Data Science
- Software Development

GitHub: **https://github.com/yourusername**

LinkedIn: **https://linkedin.com/in/yourprofile**

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork the project

🐞 Report issues

💡 Suggest new features

---

> Built with ❤️ using **Python**, **Scikit-learn**, **NLTK**, and **Streamlit**.