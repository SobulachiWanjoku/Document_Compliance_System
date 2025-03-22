import logging
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import docx
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = [para.text for para in doc.paragraphs]
        return '\n'.join(text)
    except Exception as e:
        logging.error(f"Failed to extract text from DOCX file {file_path}: {str(e)}")
        raise

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
            return '\n'.join(text)
    except Exception as e:
        logging.error(f"Failed to extract text from PDF file {file_path}: {str(e)}")
        raise

def extract_text_from_file(file_path: str) -> str:
    """Extract text from a DOCX or PDF file."""
    if not isinstance(file_path, str) or not file_path:
        logging.error("Invalid file path provided.")
        raise ValueError("File path must be a non-empty string.")
    try:
        if file_path.endswith('.docx'):
            return extract_text_from_docx(file_path)
        elif file_path.endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        else:
            raise ValueError("Unsupported file type. Only DOCX and PDF are supported.")
    except Exception as e:
        logging.error(f"Failed to extract text from file {file_path}: {str(e)}")
        raise

def preprocess_text(text: str) -> str:
    """Preprocess text for better comparison."""
    if not isinstance(text, str) or not text:
        logging.error("Invalid text provided for preprocessing.")
        raise ValueError("Text must be a non-empty string.")
    try:
        # Log original text (first 50 characters)
        logging.info(f"Original text: {text[:50]}...")
        text = text.lower()  # Convert to lowercase
        logging.info(f"Lowercased text: {text[:50]}...")
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in stop_words]
        logging.info(f"Filtered text (first 10 words): {filtered_words[:10]}...")
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        logging.info(f"Lemmatized text (first 10 words): {lemmatized_words[:10]}...")
        return ' '.join(lemmatized_words)
    except Exception as e:
        logging.error(f"Text preprocessing failed: {str(e)}")
        raise ValueError(f"Text preprocessing failed: {str(e)}")

def save_vectorizer(vectorizer: TfidfVectorizer, file_path: str):
    """Save the vectorizer to a file using pickle."""
    try:
        with open(file_path, 'wb') as file:
            pkl.dump(vectorizer, file)
        logging.info(f"Vectorizer saved to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to save vectorizer: {str(e)}")
        raise

def load_vectorizer(file_path: str) -> TfidfVectorizer:
    """Load the vectorizer from a file using pickle."""
    try:
        with open(file_path, 'rb') as file:
            vectorizer = pkl.load(file)
        logging.info(f"Vectorizer loaded from {file_path}.")
        return vectorizer
    except Exception as e:
        logging.error(f"Failed to load vectorizer: {str(e)}")
        raise

def check_compliance(template_text: str, student_path: str, threshold: float = 0.95, vectorizer=None):
    """
    Check document compliance based on cosine similarity between a template and a student document.
    Returns a tuple of (similarity_score in %, compliance_status).
    """
    if not isinstance(template_text, str) or not template_text:
        logging.error("Invalid template text provided.")
        raise ValueError("Template text must be a non-empty string.")
    if vectorizer is None:
        logging.error("No vectorizer provided for compliance check.")
        raise ValueError("A fitted TfidfVectorizer must be provided.")

    try:
        student_text = extract_text_from_file(student_path)
        # If student_text is in bytes, decode it
        if isinstance(student_text, bytes):
            student_text = student_text.decode('utf-8')
        # Preprocess both template and student texts
        template_processed = preprocess_text(template_text)
        student_processed = preprocess_text(student_text)
        # Calculate TF-IDF vectors and cosine similarity
        tfidf_matrix = vectorizer.transform([template_processed, student_processed])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = similarity[0][0]
        logging.info(f"Cosine similarity score: {score * 100:.2f}%")
        compliance_status = "Compliant" if score >= threshold else "Non-Compliant"
        return score * 100, compliance_status
    except Exception as e:
        logging.error(f"Compliance check failed: {str(e)}")
        raise ValueError(f"Compliance check failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Example template text and student document path
        template_text = "This is a sample template that guides the document structure."
        student_document_path = "path_to_student_document.docx"  # Replace with your actual file path

        # Initialize and fit the vectorizer on the preprocessed template text
        vectorizer = TfidfVectorizer(stop_words='english')
        template_processed = preprocess_text(template_text)
        vectorizer.fit([template_processed])
        
        # Optionally, save the vectorizer for future use
        save_vectorizer(vectorizer, "vectorizer.pkl")
        
        # Load the vectorizer (simulating usage in a separate session)
        loaded_vectorizer = load_vectorizer("vectorizer.pkl")
        
        # Check compliance of the student document against the template
        score, status = check_compliance(template_text, student_document_path, threshold=0.95, vectorizer=loaded_vectorizer)
        logging.info(f"Document compliance status: {status} with a score of {score:.2f}%")
    except Exception as e:
        logging.error(f"Error during document processing: {str(e)}")
