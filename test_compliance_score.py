import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unittest

# Load the pre-trained model and vectorizer
try:
    model = joblib.load('saved_models/compliance_model.pkl')
    vectorizer = joblib.load('saved_models/vectorizer.pkl')

    # Ensure the vectorizer has been fitted
    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
        raise ValueError("The loaded vectorizer has an empty vocabulary. Ensure it was trained before saving.")

except FileNotFoundError as e:
    raise FileNotFoundError("One or more required files (model/vectorizer) were not found. Ensure the correct paths.") from e

# Function to calculate compliance score
def calculate_compliance_score(template_text, student_text, vectorizer):
    """
    Computes a compliance score between a template and a student's text using TF-IDF and cosine similarity.

    Args:
        template_text (str): The template document.
        student_text (str): The student's submitted document.
        vectorizer (TfidfVectorizer): A pre-trained TF-IDF vectorizer.

    Returns:
        float: The compliance score as a percentage.
    """
    # Ensure inputs are strings (decode bytes if necessary)
    if isinstance(template_text, bytes):
        template_text = template_text.decode('utf-8')
    if isinstance(student_text, bytes):
        student_text = student_text.decode('utf-8')

    # Convert text to TF-IDF vectors
    tfidf_matrix = vectorizer.transform([template_text, student_text])
    
    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0] * 100  # Scale to percentage

# Unit test class
class TestComplianceScore(unittest.TestCase):

    def setUp(self):
        """Set up test data before each test case."""
        self.template_text = "This is a sample template text."
        self.student_text = "This is a sample template text."

    def test_string_input(self):
        """Test compliance score for identical string inputs."""
        score = calculate_compliance_score(self.template_text, self.student_text, vectorizer)
        self.assertAlmostEqual(score, 100.0, places=2)

    def test_bytes_input(self):
        """Test compliance score when input is in bytes format."""
        student_text_bytes = b"This is a sample template text."
        score_bytes = calculate_compliance_score(self.template_text, student_text_bytes, vectorizer)
        self.assertAlmostEqual(score_bytes, 100.0, places=2)

    def test_partial_match(self):
        """Test compliance score when there's a partial match."""
        partial_text = "This is a sample text."
        score = calculate_compliance_score(self.template_text, partial_text, vectorizer)
        self.assertLess(score, 100.0)
        self.assertGreater(score, 50.0)  # Assuming partial match gives a moderate score

    def test_no_match(self):
        """Test compliance score for completely different texts."""
        different_text = "Completely unrelated content."
        score = calculate_compliance_score(self.template_text, different_text, vectorizer)
        self.assertLess(score, 20.0)  # Assuming no similarity should yield a low score

# Run unit tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
