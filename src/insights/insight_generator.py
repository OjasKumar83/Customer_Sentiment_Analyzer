import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from prophet import Prophet
import os

#Loading Dataset
data_path = "data/processed/cleaned_reviews.csv"
print("Loading dataset...")
df = pd.read_csv(data_path)
texts = df['clean_text'].dropna().tolist()

#TF-IDF Vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(texts)

#Traning the model on KMeans Clustering
num_clusters = 5
print(f"Clustering into {num_clusters} groups...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

#Extracting top keywords per cluster
terms = vectorizer.get_feature_names_out()
top_keywords = {}
for i in range(num_clusters):
    cluster_center = kmeans.cluster_centers_[i]
    top_indices = cluster_center.argsort()[-10:][::-1]
    top_words = [terms[ind] for ind in top_indices]
    top_keywords[i] = top_words

def generate_summary(words):
    if "taste" in words or "flavor" in words:
        return "Complaints mainly about taste and flavor."
    elif "delivery" in words or "shipping" in words:
        return "Issues related to delivery and logistics."
    elif "price" in words or "cost" in words:
        return "Concerns about product price or value."
    elif "packaging" in words or "box" in words:
        return "Problems related to packaging and freshness."
    elif "service" in words or "support" in words:
        return "Customer service and response-related issues."
    else:
        return "General product experience feedback."

cluster_summaries = {}
for i, words in top_keywords.items():
    cluster_summaries[i] = generate_summary(words)


os.makedirs("reports/plots", exist_ok=True)
os.makedirs("reports/wordclouds", exist_ok=True)

#WordClouds genration
print("Generating wordclouds...")
for i, words in top_keywords.items():
    text = " ".join(words)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Cluster {i+1}: ~{len(df[df['cluster']==i])} complaints")
    img_path = f"reports/wordclouds/cluster_{i+1}.png"
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

#Cluster Distribution Bar Chart using pyplot
plt.figure(figsize=(6, 4))
cluster_counts = df['cluster'].value_counts().sort_index()
plt.bar(range(num_clusters), cluster_counts, color='skyblue')
plt.title("Complaint Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Reviews")
bar_chart_path = "reports/plots/cluster_distribution.png"
plt.tight_layout()
plt.savefig(bar_chart_path)
plt.close()

#Using prophet to forecast customer satisfaction score trends for next month
print("Building customer satisfaction forecast...")
if 'Score' in df.columns:
    # Create synthetic timestamp if not available
    if 'Time' in df.columns:
        df['date'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
    else:
        df['date'] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")

    daily_scores = df.groupby('date')['Score'].mean().reset_index()
    daily_scores.columns = ['ds', 'y']

    model = Prophet(daily_seasonality=True)
    model.fit(daily_scores)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    #Plotting forecast
    plt.figure(figsize=(7, 4))
    plt.plot(daily_scores['ds'], daily_scores['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
    plt.title("Customer Satisfaction Trend Forecast (Next 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Average Score")
    plt.legend()
    trend_path = "reports/plots/trend_forecast.png"
    plt.tight_layout()
    plt.savefig(trend_path)
    plt.close()
else:
    trend_path = None

#Creating PDF Report
pdf_path = "reports/AI_insights_report.pdf"
print(f"Generating PDF report: {pdf_path}")

c = canvas.Canvas(pdf_path, pagesize=letter)
width, height = letter
y = height - 80
c.setFont("Helvetica-Bold", 18)
c.drawString(50, y, "AI Insights Report")
y -= 40
c.setFont("Helvetica", 12)

#Adding cluster summaries
for i in range(num_clusters):
    c.drawString(50, y, f"Cluster {i+1}: {cluster_summaries[i]}")
    y -= 20
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(60, y, f"Top Keywords: {', '.join(top_keywords[i])}")
    y -= 40
    c.setFont("Helvetica", 12)
    if y < 150:
        c.showPage()
        y = height - 80

#Added some charts for better insights
def add_image(c, path, x, y, w=400):
    if os.path.exists(path):
        img = ImageReader(path)
        c.drawImage(img, x, y, width=w, preserveAspectRatio=True, mask='auto')

c.showPage()
c.setFont("Helvetica-Bold", 14)
c.drawString(50, height - 50, "Visual Insights")

add_image(c, bar_chart_path, 80, 350, 400)

#Add trend forecast
if trend_path:
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Predicted Customer Satisfaction Trend")
    add_image(c, trend_path, 80, 250, 400)

# wordclouds
for i in range(num_clusters):
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, f"Cluster {i+1} WordCloud")
    add_image(c, f"reports/wordclouds/cluster_{i+1}.png", 80, 250, 400)
#DELIVERABLE 4
c.save()
print(f"Report saved successfully at {pdf_path}")
