# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sample dataset (you can replace this with your CSV file)
data = {
    "email": [
        "Win a free lottery now",
        "Meeting schedule for tomorrow",
        "Claim your free prize",
        "Project discussion update",
        "Limited time offer click now",
        "Team lunch invitation"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Encode target labels (spam = 1, ham = 0)
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Features and target
X = df["email"]
y = df["label_encoded"]

# Create a pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model
scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')

# Results
print("Accuracy scores for each fold:", scores)
print("Mean accuracy:", np.mean(scores))
