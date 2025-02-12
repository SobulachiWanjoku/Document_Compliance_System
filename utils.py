import cv2
import pytesseract
import numpy as np
from docx import Document
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For loading the pre-trained model

# Load the pre-trained model (ensure the model file is available)
model = joblib.load('saved_models/compliance_model.pkl')  # Updated model path
vectorizer = joblib.load('saved_models/vectorizer.pkl')  # Load the vectorizer

def check_compliance(template_content, student_path):
    # Determine the file type based on the file extension
    file_extension = os.path.splitext(student_path)[1].lower()

    if file_extension == '.docx':
        # Handle DOCX file
        doc = Document(student_path)
        student_text = "\n".join([para.text for para in doc.paragraphs])
    else:
        # Handle image file
        student_img = cv2.imread(student_path)
        if student_img is None:
            raise ValueError(f"Failed to load student file at path: {student_path}. Ensure the file exists and the path is correct.")
        student_gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)
        student_text = pytesseract.image_to_string(student_gray)

    # Convert binary template content to text
    if isinstance(template_content, bytes):
        try:
            template_text = template_content.decode('utf-8')
        except UnicodeDecodeError:
            template_text = template_content.decode('latin-1')  # Fallback to another encoding
    else:
        template_text = template_content

    # Calculate compliance score using the machine learning model
    compliance_score = calculate_compliance_score(template_text, student_text)
    recommendations = generate_recommendations(template_text, student_text)

    return compliance_score, recommendations

def calculate_compliance_score(template_text, student_text):
    # Use TF-IDF Vectorization and Cosine Similarity for scoring
    tfidf_matrix = vectorizer.transform([template_text, student_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0] * 100  # Scale to percentage

def generate_recommendations(template_text, student_text):
    # Placeholder for recommendations logic
    return ["Check formatting", "Ensure all sections are included"]
