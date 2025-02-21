import cv2
import pytesseract
import numpy as np
from docx import Document
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load the pre-trained model and vectorizer
try:
    try:
        model = joblib.load('saved_models/compliance_model.pkl')
        vectorizer = joblib.load('saved_models/vectorizer.pkl')
    except Exception as e:
        raise ImportError(f"Failed to load model files: {str(e)}")

except Exception as e:
    raise ImportError(f"Failed to load model files: {str(e)}")



def check_compliance(template_text, student_path):
    """Check document compliance and generate recommendations"""

    """Check document compliance and generate recommendations"""
    try:
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
                raise ValueError(f"Failed to load student file at path: {student_path}")
            student_gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)
            student_text = pytesseract.image_to_string(student_gray)

        # Preprocess texts
        template_text = preprocess_text(template_text)
        student_text = preprocess_text(student_text)

        # Calculate similarity
        tfidf_matrix = vectorizer.transform([template_text, student_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = similarity[0][0] * 100  # Scale to percentage

        # Generate recommendations
        recommendations = generate_recommendations(template_text, student_text)
        
        return score, recommendations

    except Exception as e:
        raise ValueError(f"Compliance check failed: {str(e)}")


def generate_recommendations(template_text, student_text):
    """Generate specific recommendations based on text comparison"""
    recommendations = []
    
    # Check for missing sections
    template_sections = set(re.findall(r'\[SECTION: (.*?)\]', template_text))
    student_sections = set(re.findall(r'\[SECTION: (.*?)\]', student_text))
    missing_sections = template_sections - student_sections
    if missing_sections:
        recommendations.append(f"Missing sections: {', '.join(missing_sections)}")

    # Check formatting issues
    template_format = re.findall(r'\[FORMAT: (.*?)\]', template_text)
    student_format = re.findall(r'\[FORMAT: (.*?)\]', student_text)
    if template_format != student_format:
        recommendations.append("Formatting does not match template requirements")

    # Check for minimum word count
    template_word_count = len(template_text.split())
    student_word_count = len(student_text.split())
    if student_word_count < 0.8 * template_word_count:
        recommendations.append(f"Document is too short (expected at least {int(0.8 * template_word_count)} words)")

    return recommendations if recommendations else ["Document meets all requirements"]

def preprocess_text(text):
    """Preprocess text for better comparison"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    return ' '.join(lemmatized_words)
