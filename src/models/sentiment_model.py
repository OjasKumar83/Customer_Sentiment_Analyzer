'''Having CUDA enabled pytorch will make the traning 10 to 20 times faster as it will also use GPU(if present) during the traininig along with cpu
   Run this command in pip to do so: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ''' 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW #It is working fine, its just not Pylance friendly
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import joblib
import os

#Loading preprocessed dataset
data_path = "data/processed/cleaned_reviews.csv"
print("Loading cleaned dataset...")
df = pd.read_csv(data_path)

#Label Encoding
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['label'] = df['Sentiment'].map(label_map)

#Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#Tokenizing
def tokenize_function(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=128)

print("Tokenizing data...")
train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

class SentimentDataset(torch.utils.data.Dataset): #this is working fine, its just not Pylance friendly
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, list(train_labels))
test_dataset = SentimentDataset(test_encodings, list(test_labels))

#DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#Model
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

#Moving model to correct device
model = model.to(device)  # type: ignore

optimizer = AdamW(model.parameters(), lr=3e-5)  #

#Trainnig in loops
epochs = 1
model.train()
print("Training started...")

for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):  
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

#Evaluation
model.eval()
preds, true_labels = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

acc = accuracy_score(true_labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

print(f"\nEvaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

#Saving teh  model as pkl file #DELIVERABLE 2
os.makedirs("models", exist_ok=True)
model_save_path = "models/sentiment_model.pkl"
joblib.dump(model, model_save_path)
print(f"\nModel saved successfully to {model_save_path}")
