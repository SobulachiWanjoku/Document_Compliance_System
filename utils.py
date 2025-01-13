import cv2
import pytesseract
import numpy as np

def check_compliance(template_content, student_path):
    # Convert binary template content to an image
    nparr = np.frombuffer(template_content, np.uint8)
    template_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Load student image
    student_img = cv2.imread(student_path)

    # Convert images to grayscale
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    student_gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)

    # Use OCR to extract text
    template_text = pytesseract.image_to_string(template_gray)
    student_text = pytesseract.image_to_string(student_gray)

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