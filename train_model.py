import os
import joblib
import numpy as np
import re
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Ensure necessary NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sample dataset (Consider expanding this for better performance)
data = [
    ("This is the official template", "This is the official template", 1),
    ("This is the official template", "This is a different document", 0),
    ("The report should include findings", "The report includes all findings", 1),
    ("The report should include findings", "Introduction and references only", 0),
    ("Ensure all sections are covered", "All sections are covered as required", 1),
    ("Ensure all sections are covered", "The conclusion is missing", 0),
    ("Include references in APA format", "References are formatted correctly", 1),
    ("Include references in APA format", "No references included", 0),
    ("Methods must be described", "Methods are clearly described", 1),
    ("Methods must be described", "Only results are mentioned", 0)
]

# Text preprocessing function
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters except spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Extract template, student texts, and labels
template_texts = [preprocess_text(item[0]) for item in data]
student_texts = [preprocess_text(item[1]) for item in data]
y = np.array([item[2] for item in data])

# Print label distribution before splitting
print("Label distribution before split:", Counter(y))

# Feature extraction and model pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # TF-IDF for feature extraction
    ('classifier', LogisticRegression(class_weight='balanced', C=1.0, solver='liblinear'))
])

# Split data using stratified sampling to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    student_texts, y, test_size=0.2, random_state=42, stratify=y
)

try:
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and pipeline
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(save_dir, 'compliance_model.pkl'))
    print("Model training complete and saved.")

except Exception as e:
    print(f"An error occurred during training: {e}")
