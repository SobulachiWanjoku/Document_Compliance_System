import cv2
import pytesseract
import numpy as np
from docx import Document
import os

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

    # Simple compliance check (you can enhance this logic)
    compliance_score = calculate_compliance_score(template_text, student_text)
    recommendations = generate_recommendations(template_text, student_text)

    return compliance_score, recommendations

def calculate_compliance_score(template_text, student_text):
    # Basic score calculation (you can enhance this logic)
    if template_text.strip() == student_text.strip():
        return 100
    else:
        return 50  # Placeholder score

def generate_recommendations(template_text, student_text):
    # Placeholder for recommendations logic
    return ["Check formatting", "Ensure all sections are included"]
