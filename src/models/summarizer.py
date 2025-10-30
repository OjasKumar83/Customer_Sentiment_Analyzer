import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from tqdm import tqdm
import os

#Loading summarization model
print("Loading T5 summarization model...")
model_name = "t5-small"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

#Device setup (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Loading dataset
data_path = "data/processed/cleaned_reviews.csv"
print(f"Loading dataset from {data_path} ...")
df = pd.read_csv(data_path)

#We'll summarize the 'clean_text' column
texts = df['clean_text'].tolist()[:5]  # limit to 5 examples for demo

#Function to generate short and detailed summaries
def generate_summary(text, summary_type="short"):
    prefix = "summarize: " + text
    inputs = tokenizer.encode(prefix, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    if summary_type == "short":
        max_len, min_len = 50, 10
    else:  # detailed
        max_len, min_len = 120, 40

    summary_ids = model.generate(
        inputs,
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#Generate summaries for each review
results = []
print("Generating summaries...\n")
for text in tqdm(texts):
    short_sum = generate_summary(text, summary_type="short")
    detailed_sum = generate_summary(text, summary_type="detailed")
    results.append({
        "Original Text": text,
        "Short Summary": short_sum,
        "Detailed Summary": detailed_sum
    })

print("\nGenerated Summaries:\n") #DELIVERABLE 3
for i, r in enumerate(results, 1):
    print(f"\nReview {i}:")
    print(f"Original Text: {r['Original Text'][:200]}...")
    print(f"Short Summary: {r['Short Summary']}")
    print(f"Detailed Summary: {r['Detailed Summary']}")
    print("-" * 80)

#Save summaries to a file
import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/generated_summaries.csv"
pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8')

print(f"\nSummaries saved to {output_path}")