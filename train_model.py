import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter  # Import Counter to check label distribution

# Sample dataset (ensure a mix of 0 and 1 labels)
data = [
    ("Template text 1", "Student submission 1", 1),
    ("Template text 2", "Student submission 2", 0),
    ("Template text 3", "Student submission 3", 1),
    ("Template text 4", "Student submission 4", 0),
    ("Template text 5", "Student submission 5", 1),
    ("Template text 6", "Student submission 6", 0),
]

# Extract student texts and labels
student_texts = [item[1] for item in data]
y = np.array([item[2] for item in data])  # Define y before using it

# Print label distribution before splitting
print("Label distribution before split:", Counter(y))

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(student_texts)

# Split data using stratified sampling to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print label distribution after splitting
print("Label distribution in training set:", Counter(y_train))
print("Label distribution in test set:", Counter(y_test))

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

joblib.dump(model, os.path.join(save_dir, 'compliance_model.pkl'))
joblib.dump(vectorizer, os.path.join(save_dir, 'vectorizer.pkl'))

print("Model training complete and saved.")
