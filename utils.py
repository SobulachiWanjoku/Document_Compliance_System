import os
import cv2
import pytesseract
import re
import joblib
import logging
import nltk
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import chardet

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model and vectorizer
try:
    model = joblib.load('saved_models/compliance_model.pkl')
    vectorizer = joblib.load('saved_models/vectorizer.pkl')
except Exception as e:
    logging.error(f"Failed to load model files: {str(e)}")
    raise ImportError(f"Failed to load model files: {str(e)}. Ensure files exist and are properly trained.")

def preprocess_text(text):
    """Preprocess text for better comparison"""
    try:
        text = text.lower()  # Convert to lowercase

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words]

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

        return ' '.join(lemmatized_words)

    except Exception as e:
        logging.error(f"Text preprocessing failed: {str(e)}")
        raise ValueError(f"Text preprocessing failed: {str(e)}")

def extract_text_from_file(file_path):
    """Extracts text from either a .docx file or an image file"""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.docx':
            doc = Document(file_path)
            extracted_text = "\n".join([para.text for para in doc.paragraphs])

        elif file_extension in ['.jpg', '.jpeg', '.png']:
            student_img = cv2.imread(file_path)
            if student_img is None:
                raise ValueError(f"Failed to load image at path: {file_path}")
            student_gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)
            extracted_text = pytesseract.image_to_string(student_gray).strip()

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Detect file encoding before decoding
        raw_bytes = extracted_text.encode(errors='replace')
        detected_encoding = chardet.detect(raw_bytes)['encoding']
        
        # Attempt to decode using detected encoding
        try:
            return raw_bytes.decode(detected_encoding, errors='replace')
        except (UnicodeDecodeError, TypeError):
            # Fallback to 'latin-1' encoding if detection fails
            return raw_bytes.decode('latin-1', errors='replace')


    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {str(e)}")
        raise ValueError(f"Error extracting text from file: {str(e)}")
    
def check_compliance(template_text, student_path):
    """Check document compliance and generate recommendations"""
    try:
        logging.info("Extracting text from student file...")
        logging.info(f"Template text: {template_text[:50]}...")  # Log the first 50 characters of the template text

        student_text = extract_text_from_file(student_path)

        # Ensure student_text is a string
        if isinstance(student_text, bytes):
            student_text = student_text.decode('utf-8')

        logging.info("Preprocessing texts...")
        logging.info(f"Student text: {student_text[:50]}...")  # Log the first 50 characters of the student text

        template_text = preprocess_text(template_text)
        student_text = preprocess_text(student_text)

        logging.info("Calculating similarity...")
        logging.info("TF-IDF matrix created successfully.")

        tfidf_matrix = vectorizer.transform([template_text, student_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = round(similarity[0][0] * 100, 2)  # Scale to percentage

        logging.info("Generating recommendations...")
        logging.info(f"Score calculated: {score}")  # Log the calculated score

        recommendations = generate_recommendations(template_text, student_text)
        
        return score, recommendations

    except Exception as e:
        logging.error(f"Compliance check failed: {str(e)}")
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

def save_template(name, path, db, Template):
    """Save template details to the database"""
    try:
        # Extract text before saving
        extracted_text = extract_text_from_file(path)
        
        # Save template path and extracted content
        new_template = Template(name=name, path=path, content=extracted_text)  
        db.session.add(new_template)
        db.session.commit()
        print("Template saved successfully.")
    
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error saving template: {str(e)}")
        print(f"Error saving template: {str(e)}")
