import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset

data = pd.read_csv('../data/phishing_site_urls.csv')

# Rename columns if needed

data.columns = ['label', 'url']

# Convert labels (bad=1, good=0)

data['label'] = data['label'].map({'bad':1, 'good':0})

# Features and labels

X = data['url']
y = data['label']

# Convert text to numbers

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2)

# Train model

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model

with open('../models/model.pkl', 'wb') as f:
pickle.dump(model, f)

with open('../models/vectorizer.pkl', 'wb') as f:
pickle.dump(vectorizer, f)

print("Model trained and saved!")
