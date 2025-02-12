import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model and vectorizer
model = joblib.load('saved_models/compliance_model.pkl')
vectorizer = joblib.load('saved_models/vectorizer.pkl')

def calculate_compliance_score(template_text, student_text, vectorizer):
    # Use TF-IDF Vectorization and Cosine Similarity for scoring
    tfidf_matrix = vectorizer.transform([template_text, student_text])
    print("TF-IDF Matrix:\n", tfidf_matrix.toarray())  # Debugging statement
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    print("Similarity Score:\n", similarity)  # Debugging statement
    return similarity[0][0] * 100  # Scale to percentage

# Test case
template_text = "This is a sample template text."
student_text = "This is a sample template text."

# Calculate compliance score
score = calculate_compliance_score(template_text, student_text, vectorizer)

print(f"Compliance Score: {score:.2f}%")
