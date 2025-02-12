import os
import joblib
import numpy as np
import re
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import nltk 
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample dataset (Consider expanding this for better performance)
data = [
    ("Template text 1", "Student submission 1", 1),
    ("Template text 2", "Student submission 2", 0),
    ("Template text 3", "Student submission 3", 1),
    ("Template text 4", "Student submission 4", 0),
    ("Template text 5", "Student submission 5", 1),
    ("Template text 6", "Student submission 6", 0),
    ("Template text 11", "Student submission 11", 1),
    ("Template text 12", "Student submission 12", 0),
    ("Template text 13", "Student submission 13", 1),
    ("Template text 14", "Student submission 14", 0)
]

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Extract student texts and labels
student_texts = [preprocess_text(item[1]) for item in data]
y = np.array([item[2] for item in data])

# Print label distribution before splitting
print("Label distribution before split:", Counter(y))

# Feature extraction and model pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(class_weight='balanced', C=1.0, solver='liblinear'))
])

# Split data using stratified sampling to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    student_texts, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and pipeline
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
joblib.dump(pipeline, os.path.join(save_dir, 'compliance_model.pkl'))

print("Model training complete and saved.")