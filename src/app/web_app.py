import streamlit as st
import pandas as pd
import torch
import os
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    T5Tokenizer, 
    T5ForConditionalGeneration
)
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# App Config
st.set_page_config(page_title="Customer Sentiment Dashboard", layout="wide")
st.title("Customer Feedback Analysis Dashboard")
st.markdown("AI-powered Sentiment Analysis, Summarization, and Insights.")

# Loading Models
@st.cache_resource
def load_models():
    sentiment_model_path = "src/models/sentiment_model.pkl"

    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        st.success("Sentiment model loaded successfully.")
    except Exception as e:
        st.warning(f"Could not load custom model, using default: {e}")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Loading summarization model
    summarizer_name = "t5-small"
    sum_tokenizer = T5Tokenizer.from_pretrained(summarizer_name)
    sum_model = T5ForConditionalGeneration.from_pretrained(summarizer_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sum_model = sum_model.to(device)

    return tokenizer, model, sum_tokenizer, sum_model, device

sent_tokenizer, sent_model, sum_tokenizer, sum_model, device = load_models()

# Helper Functions
def predict_sentiment(text):
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = sent_model(**inputs)
        pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(pred, dim=1).item()
    return "Positive" if label == 1 else "Negative"

def summarize_text(text, summary_type="short"):
    prefix = "summarize: " + text
    inputs = sum_tokenizer.encode(prefix, return_tensors="pt", max_length=512, truncation=True).to(device)
    if summary_type == "short":
        max_len, min_len = 50, 10
    else:
        max_len, min_len = 120, 40

    summary_ids = sum_model.generate(
        inputs, max_length=max_len, min_length=min_len, num_beams=4, early_stopping=True
    )
    return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Data Loading Sectio
default_path = "data/processed/cleaned_reviews.csv"
st.sidebar.header("Data Input")
st.sidebar.markdown("You can upload a new CSV or use the preprocessed one from the project folder.")

use_default = st.sidebar.checkbox("Use preprocessed dataset (cleaned_reviews.csv)", value=True)

if use_default and os.path.exists(default_path):
    df = pd.read_csv(default_path)
    st.sidebar.success("Loaded cleaned_reviews.csv from project data.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded custom CSV.")
    else:
        st.warning("No data selected yet. Upload a file or use the preprocessed dataset.")
        st.stop()

if "clean_text" not in df.columns:
    st.error("CSV must contain a column named 'clean_text'. Please check your file.")
    st.stop()

# Sentiment Analysis
st.subheader("Sentiment Analysis")
with st.spinner("Analyzing sentiments..."):
    df["Predicted Sentiment"] = df["clean_text"].apply(predict_sentiment)
st.success("Sentiment analysis completed!")
st.dataframe(df[["clean_text", "Predicted Sentiment"]].head())

# Summarization
st.subheader("AI Summaries")
sample_df = df.head(5).copy()
with st.spinner("Generating summaries for sample reviews..."):
    sample_df["Short Summary"] = sample_df["clean_text"].apply(lambda x: summarize_text(x, "short"))
    sample_df["Detailed Summary"] = sample_df["clean_text"].apply(lambda x: summarize_text(x, "detailed"))

st.table(sample_df[["clean_text", "Short Summary", "Detailed Summary", "Predicted Sentiment"]])

# Visualization
st.subheader("Sentiment Insights")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Predicted Sentiment", palette="viridis", ax=ax)
plt.title("Sentiment Distribution")
st.pyplot(fig)

# Pie chart
sentiment_counts = df["Predicted Sentiment"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(sentiment_counts, labels=sentiment_counts.index.tolist(), autopct="%1.1f%%", startangle=90, colors=["#4CAF50", "#F44336"])
ax2.axis("equal")
st.pyplot(fig2)

# Download Section
st.subheader("Export Results")
buffer = BytesIO()
df.to_csv(buffer, index=False)
buffer.seek(0)
st.download_button("Download Processed Results (CSV)", data=buffer, file_name="final_sentiment_results.csv", mime="text/csv")
