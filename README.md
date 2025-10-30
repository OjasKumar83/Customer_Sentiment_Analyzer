# Customer Sentiment Analyzer

The Customer Sentiment Analyzer is a machine learning web application that predicts the sentiment of customer reviews. It helps businesses understand whether customer feedback is positive, negative, or neutral.

# Features

Sentiment Prediction: Analyzes customer reviews and classifies them as Positive, Negative, or Neutral.

Preprocessing Pipeline: Cleans and tokenizes text using NLP techniques such as stopword removal, lemmatization, and punctuation filtering.

Machine Learning Model: Uses a trained model (Logistic Regression / SVM) with TF-IDF vectorization for text classification.

Interactive Web Interface: Built using Streamlit, allowing users to input reviews and instantly view predictions.

Modular Code Design: Structured into separate modules for data preprocessing, model training, and prediction.

Scalable and Reusable: Easily adaptable to other text-based sentiment or feedback analysis tasks.

# Tech Stack

Python 3.11

Streamlit for web deployment

Scikit-learn for model training

Pandas & NumPy for data handling

NLTK for text preprocessing

# Project Structure

Customer_Sentiment_Analysis/
│

├── app.py                     # Streamlit app entry point

├── models/
│   └── sentiment_model.pkl    # Trained sentiment analysis model

├── data/
│   └── raw/Reviews.csv        # Dataset used for training (not included in repo)

├── utils/
│   ├── preprocessing.py       # Text cleaning and preprocessing functions

│   └── model_utils.py         # Model loading and prediction utilities

├── requirements.txt           # Python dependencies

└── README.md                  # Project documentation


# How It Works

The input text (customer review) is cleaned and preprocessed using NLP techniques.

The processed text is transformed into numerical form using TF-IDF.

The trained machine learning model predicts the sentiment category.

The result is displayed on the Streamlit web interface.

