import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download NLTK dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # if issue with punkt_tab not found occurs please install it directly using nltk.download('punkt_tab') in your virtual environment
nltk.download('wordnet')

#Loading Dataset
file_path = "data/raw/Reviews.csv"
print("📂 Loading dataset...")
df = pd.read_csv(file_path)

#Keep only the columns needed
df = df[['Score', 'Summary', 'Text']]

# Remove rows with missing text
df.dropna(subset=['Text'], inplace=True)

#Creating labels
def get_sentiment(score):
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['Sentiment'] = df['Score'].apply(get_sentiment)

#Data Cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)        
    text = re.sub(r'http\S+|www\S+', '', text)  
    text = re.sub(r'[^a-z\s]', '', text)     
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

tqdm.pandas(desc="Cleaning text")
df['clean_text'] = df['Text'].progress_apply(clean_text)

#Dwonsampling the datset for quikcer trainnig purposes to 5000 instances
df = df.sample(n=5000, random_state=42)

import os

#Saving teh cleaned dataset
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True) 

output_path = os.path.join(output_dir, "cleaned_reviews.csv")
df.to_csv(output_path, index=False)

print(f"Cleaned dataset with {len(df)} records saved to {output_path}")
print(df.head())
