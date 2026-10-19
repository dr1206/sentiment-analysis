# Tweet Sentiment Analysis using Machine Learning

## Overview
This project is a **web-based sentiment analysis application** that classifies English tweets as **Positive** or **Negative**. It uses the **Sentiment140 dataset** (1.6 million labeled tweets) and applies multiple machine learning algorithms to provide predictions.

The app demonstrates how **natural language processing (NLP)** and **machine learning (ML)** can be applied to social media data for sentiment classification.

---

## Features
- Analyze the sentiment of any English tweet.
- Supports hashtags, mentions, emojis, and informal tweet language.
- Provides predictions from **three models**:
  - Bernoulli Naive Bayes
  - Linear SVM
  - Logistic Regression
- **Final prediction** is calculated using a majority vote across models.
- Word count is displayed for each tweet.
- Clean, dark metallic UI for a smart user experience.

---

## Dataset
- **Sentiment140 dataset**  
- Contains **1.6 million tweets** with the following key columns:
  1. **Polarity:** 0 = Negative, 4 = Positive
  2. **Text:** The actual tweet content
- Neutral tweets (polarity = 2) are removed for **binary classification**.
- Tweets are cleaned (lowercased, URLs and mentions removed) while keeping punctuation, hashtags, and emojis for accurate sentiment detection.

---

## Technical Stack
- **Backend:** Python, Flask
- **Frontend:** HTML, CSS
- **Machine Learning:** scikit-learn
- **Data Handling:** pandas
- **Model Storage:** pickle

---

## Algorithm
1. Preprocess the tweet text:
   - Lowercase
   - Remove URLs and mentions
   - Keep punctuation, hashtags, and emojis
2. Convert text to **TF-IDF vectors** (max 5000 features, unigram + bigram)
3. Train three ML models:
   - Bernoulli Naive Bayes
   - Linear SVM
   - Logistic Regression
4. Make predictions for new tweets.
5. Calculate **final sentiment** using majority vote.

---
## Demo

Check out the live hosted application here:  
[Sentiment Analysis Web App](https://sentimentanalysis-o3kh.onrender.com)

---
## Installation
1. Clone the repository:
```bash
git clone https://github.com/athira-saji/sentiment-analysis.git
cd sentiment-analysis
