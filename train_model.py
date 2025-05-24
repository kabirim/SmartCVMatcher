import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import yaml
import numpy as np
from xgboost import XGBClassifier
import requests

# Load the YAML config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name_Bert_transformer = config["model"]["name"]

objective = config["training"]["objective"]
eval_metric = config["training"]["eval_metric"]
use_label_encoder = config["training"]["use_label_encoder"]
n_estimators = config["training"]["n_estimators"]
max_depth = config["training"]["max_depth"]
learning_rate = config["training"]["learning_rate"]
subsample = config["training"]["subsample"]
colsample_bytree = config["training"]["colsample_bytree"]

test_size = config["data"]["test_size"]
random_seed = config["data"]["random_seed"]

train_url = config["urls"]["train_url"]
test_url = config["urls"]["test_url"]

# Download and save locally
with open("train.csv", "wb") as f:
    f.write(requests.get(train_url).content)

with open("test.csv", "wb") as f:
    f.write(requests.get(test_url).content)

print("✅ Files uploaded successfully")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

df = pd.concat([train_df, test_df], ignore_index=True)

# Delete incomplete lines
df.dropna(subset=["resume_text", "job_description_text", "label"], inplace=True)

train, test = train_test_split(df, test_size=test_size, random_state=random_seed, stratify=df["label"])
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

label_mapping = {
    "No Fit": 0,
    "Potential Fit": 0,  # Optional, depending on logic
    "Good Fit": 1
}
train["label"] = train["label"].map(label_mapping)
test["label"] = test["label"].map(label_mapping)

train["label"] = train["label"].astype(int)
test["label"] = test["label"].astype(int)

# 3. Sentence-BERT encoder
# A pre-trained Sentence Transformers model, based on BERT but optimized to produce sentence vectors (embeddings)
# "all-MiniLM-L6-v2" is a lightweight, fast, yet high-performance model for encoding short texts
# It transforms a text into a 384-dimensional vector (this is its "embedding size")
model_name = model_name_Bert_transformer # Embedding dimension = 768
# Create a sentence encoder, an encoder object capable of converting text into numeric vectors usable by machine learning models
encoder = SentenceTransformer(model_name)

# Encoding summaries and descriptions
resume_embeds = encoder.encode(train["resume_text"].tolist())
job_embeds = encoder.encode(train["job_description_text"].tolist())

# Horizontal concatenation
X_train = np.hstack((resume_embeds, job_embeds))  # forme : (n_samples, 1536)
y_train = train["label"].astype(int).values  # binary labels (0 or 1)

# Same for testing
resume_embeds_test = encoder.encode(test["resume_text"].tolist())
job_embeds_test = encoder.encode(test["job_description_text"].tolist())
X_test = np.hstack((resume_embeds_test, job_embeds_test))
y_test = test["label"].astype(int).values

model = XGBClassifier(
    objective=objective, # Specifies a binary classification problem with an output between 0 and 1 (logistic)
    eval_metric=eval_metric,
    use_label_encoder=use_label_encoder,
    n_estimators=n_estimators, # Total number of trees (more = more capacity but also risk of overfitting)
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample, # At each iteration, XGBoost uses 80% of the data (useful to avoid overfitting)
    colsample_bytree=colsample_bytree, # Each tree will use 80% of the columns/features (again to avoid overfitting)
    random_state=random_seed # Random seed for reproducibility (results will be identical on each run)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "xgboost_model.joblib")
joblib.dump(encoder, "sentence_encoder.joblib")

print("✅ XGBoost model + encoder saved")