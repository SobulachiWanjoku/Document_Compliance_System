import numpy as np
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import docx  # python-docx for Word documents
from typing import Dict, List
import logging
from pathlib import Path

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

logging.basicConfig(level=logging.INFO)

class DocumentComplianceAnalyzer:
    compliance_threshold = 0.7  # Default compliance score threshold

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            norm='l2'
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from a Word document."""
        if not Path(docx_path).is_file():
            logging.error(f"File not found: {docx_path}")
            return ""

        """Extract text from a Word document."""
        try:
            doc = docx.Document(docx_path)
            return " ".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logging.error(f"Error extracting text from {docx_path}: {e}")
            return ""

    def extract_headings_from_docx(self, docx_path: str) -> List[str]:
        """Extract headings from a Word document based on bold and capitalization."""
        try:
            doc = docx.Document(docx_path)
            headings = [para.text for para in doc.paragraphs if para.style.name.startswith("Heading")]
            return headings
        except Exception as e:
            logging.error(f"Error extracting headings from {docx_path}: {e}")
            return []

    def extract_formatting_from_docx(self, docx_path: str) -> Dict:
        """Extract formatting information from a Word document."""
        try:
            doc = docx.Document(docx_path)
            formatting = {
                "fonts": set(),
                "sizes": set(),
                "bold_count": 0,
                "italic_count": 0,
                "underline_count": 0,
                "alignments": set()
            }
            for para in doc.paragraphs:
                for run in para.runs:
                    if run.font:
                        formatting["fonts"].add(run.font.name or "Unknown")
                        formatting["sizes"].add(run.font.size or 0)
                        if run.bold:
                            formatting["bold_count"] += 1
                        if run.italic:
                            formatting["italic_count"] += 1
                        if run.underline:
                            formatting["underline_count"] += 1
                if para.alignment is not None:
                    formatting["alignments"].add(para.alignment)
            return formatting
        except Exception as e:
            logging.error(f"Error extracting formatting from {docx_path}: {e}")
            return {}

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text while keeping meaningful punctuation."""
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = word_tokenize(text)
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(processed_tokens)

    def calculate_similarity(self, doc1: str, doc2: str) -> float:
        """Compute textual similarity using pre-fitted TF-IDF vectorizer."""
        if not hasattr(self.vectorizer, "vocabulary_"):
            logging.error("TF-IDF vectorizer is not fitted. Please fit the vectorizer before using it.")
            return 0.0

        """Compute textual similarity using pre-fitted TF-IDF vectorizer."""
        doc1_processed = self.preprocess_text(doc1)
        doc2_processed = self.preprocess_text(doc2)
        logging.info(f"Template preprocessed text: {doc1_processed[:100]}...")  # Log first 100 characters
        logging.info(f"Student preprocessed text: {doc2_processed[:100]}...")  # Log first 100 characters
        tfidf_matrix = self.vectorizer.transform([doc1_processed, doc2_processed])
        similarity_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        logging.info(f"Cosine similarity score: {similarity_score}")
        return similarity_score

    def evaluate_compliance(self, student_doc: str, template_doc: str, student_format: Dict, template_format: Dict, student_headings: List[str], template_headings: List[str]) -> Dict:
        """Evaluate compliance based on text similarity, structure, and formatting."""
        if not student_doc or not template_doc:
            logging.error("Student document or template document is empty.")
            return {}

        # First check if documents are exactly the same file
        if student_doc == template_doc:
            return {
                "text_similarity": 1.0,
                "heading_compliance": 1.0,
                "formatting_compliance": 1.0,
                "final_compliance_score": 1.0,
                "is_compliant": True,
                "recommendations": []
            }

        similarity_score = self.calculate_similarity(student_doc, template_doc) if student_doc and template_doc else 0.0
        
        # If content is identical (>99% similarity), give full marks for content and headings
        if similarity_score > 0.99:
            heading_match = 1.0
            formatting_score = (len(student_format["fonts"].intersection(template_format["fonts"])) / max(1, len(template_format["fonts"])) +
                              len(student_format["sizes"].intersection(template_format["sizes"])) / max(1, len(template_format["sizes"])) +
                              (1 if student_format["alignments"] == template_format["alignments"] else 0)) / 3
            final_score = 0.4 + (formatting_score * 0.3) + 0.3  # Full marks for content and headings
        else:
            heading_match = len(set(student_headings).intersection(set(template_headings))) / max(1, len(template_headings))
            font_overlap = len(student_format["fonts"].intersection(template_format["fonts"])) / max(1, len(template_format["fonts"]))
            size_overlap = len(student_format["sizes"].intersection(template_format["sizes"])) / max(1, len(template_format["sizes"]))
            alignment_match = student_format["alignments"] == template_format["alignments"]
            formatting_score = (font_overlap + size_overlap + (1 if alignment_match else 0)) / 3
            final_score = (similarity_score * 0.4) + (formatting_score * 0.3) + (heading_match * 0.3)
        
        # Log formatting and heading comparison results
        logging.info(f"Template headings: {template_headings}")
        logging.info(f"Student headings: {student_headings}")
        logging.info(f"Template formatting: {template_format}")
        logging.info(f"Student formatting: {student_format}")

        # Generate recommendations based on compliance gaps
        recommendations = []
        
        if similarity_score < 0.5:
            recommendations.append("Improve content similarity with the template")
        elif similarity_score < 0.7:
            recommendations.append("Content could better match the template")
            
        if heading_match < 0.5:
            recommendations.append("Add missing headings from the template")
        elif heading_match < 0.7:
            recommendations.append("Some headings don't match the template")
            
        if formatting_score < 0.5:
            recommendations.append("Improve formatting to match the template")
        elif formatting_score < 0.7:
            recommendations.append("Some formatting doesn't match the template")

        return {
            "text_similarity": similarity_score,
            "heading_compliance": heading_match,
            "formatting_compliance": formatting_score,
            "final_compliance_score": final_score,
            "is_compliant": final_score >= self.compliance_threshold,
            "recommendations": recommendations
        }

    def save_model(self, vectorizer_path: str):
        """Save the vectorizer to disk."""
        with open(vectorizer_path, 'wb') as vec_file:
            pkl.dump(self.vectorizer, vec_file)

    def load_model(self, vectorizer_path: str):
        """Load the vectorizer from disk."""
        with open(vectorizer_path, 'rb') as vec_file:
            self.vectorizer = pkl.load(vec_file)

    def fit_vectorizer(self, corpus: List[str]):
        """Fit the vectorizer to a corpus of documents."""
        self.vectorizer.fit(corpus)

# Example usage
if __name__ == "__main__":
    analyzer = DocumentComplianceAnalyzer()

    # Fit the vectorizer with the template document and save it
    template_path = Path("C:/Users/USER/Documents/submission guideline.docx")
    template_text = analyzer.extract_text_from_docx(template_path)
    analyzer.fit_vectorizer([template_text])  # Fit vectorizer to the template
    analyzer.save_model("vectorizer.pkl")  # Save the fitted vectorizer

    # Later, load the saved vectorizer and use it for evaluation
    analyzer.load_model("vectorizer.pkl")  # Load the saved vectorizer
    if not hasattr(analyzer.vectorizer, "vocabulary_"):
            logging.error("TF-IDF vectorizer is not fitted. Please fit the vectorizer before using it.")
            exit(1)



    student_path = Path("C:/Users/USER/Documents/Quantum Machine Learning Advancing ICT for Sustainable Development.docx")
    student_text = analyzer.extract_text_from_docx(student_path)
    template_format = analyzer.extract_formatting_from_docx(template_path)
    student_format = analyzer.extract_formatting_from_docx(student_path)
    template_headings = analyzer.extract_headings_from_docx(template_path)
    student_headings = analyzer.extract_headings_from_docx(student_path)
    
    compliance_result = analyzer.evaluate_compliance(student_text, template_text, student_format, template_format, student_headings, template_headings)
    
    print(f"Final Compliance Score: {compliance_result['final_compliance_score']:.2f}")
    print(f"Document is {'Compliant' if compliance_result['is_compliant'] else 'Non-Compliant'}")
