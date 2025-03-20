import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from typing import List, Tuple
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {str(e)}")

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')

try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')


class DocumentSimilarityAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            norm='l2'
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text with advanced cleaning and normalization.
        """
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize while preserving sentence structure
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)

    def calculate_similarity(self, doc1: str, doc2: str) -> dict:
        """
        Calculate similarity between two documents with detailed metrics.
        """
        # Preprocess both documents
        doc1_processed = self.preprocess_text(doc1)
        doc2_processed = self.preprocess_text(doc2)
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform([doc1_processed, doc2_processed])
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Get feature names (terms)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get important terms from both documents
        doc1_terms = self._get_important_terms(tfidf_matrix[0], feature_names)
        doc2_terms = self._get_important_terms(tfidf_matrix[1], feature_names)
        
        # Calculate shared terms
        shared_terms = set(doc1_terms).intersection(set(doc2_terms))
        
        return {
            'similarity_score': float(similarity_score),
            'similarity_percentage': float(similarity_score * 100),
            'shared_terms': list(shared_terms),
            'doc1_unique_terms': list(set(doc1_terms) - shared_terms),
            'doc2_unique_terms': list(set(doc2_terms) - shared_terms),
            'analysis': self._generate_analysis(similarity_score)
        }

    def _get_important_terms(self, tfidf_vector, feature_names: np.ndarray, top_n: int = 10) -> List[str]:
        """
        Get the most important terms from a TF-IDF vector.
        """
        # Convert sparse matrix to dense array
        dense_vector = tfidf_vector.toarray()[0]
        
        # Get indices of top N values
        top_indices = dense_vector.argsort()[-top_n:][::-1]
        
        # Return corresponding terms
        return [feature_names[i] for i in top_indices if dense_vector[i] > 0]

    def _generate_analysis(self, similarity_score: float) -> str:
        """
        Generate a human-readable analysis of the similarity score.
        """
        if similarity_score >= 0.9:
            return "The documents are nearly identical"
        elif similarity_score >= 0.7:
            return "The documents are very similar"
        elif similarity_score >= 0.5:
            return "The documents share significant content"
        elif similarity_score >= 0.3:
            return "The documents have some similarities"
        else:
            return "The documents are substantially different"

    def batch_compare(self, documents: List[str]) -> pd.DataFrame:
        """
        Compare multiple documents and return a similarity matrix.
        """
        # Preprocess all documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Convert to DataFrame for better visualization
        return pd.DataFrame(
            similarity_matrix,
            index=[f"Doc {i+1}" for i in range(len(documents))],
            columns=[f"Doc {i+1}" for i in range(len(documents))]
        )

# Example usage
if __name__ == "__main__":
    analyzer = DocumentSimilarityAnalyzer()
    
    # Example documents
    doc1 = """
    The quick brown fox jumps over the lazy dog.
    This is a common pangram used in typography.
    """
    
    doc2 = """
    The fast brown fox leaps over the sleepy dog.
    This pangram is often used in font displays.
    """
    
    # Calculate similarity with detailed analysis
    result = analyzer.calculate_similarity(doc1, doc2)
    
    print(f"Similarity Score: {result['similarity_percentage']:.2f}%")
    print(f"Analysis: {result['analysis']}")
    print("\nShared Terms:", ', '.join(result['shared_terms']))
    print("\nUnique to Doc1:", ', '.join(result['doc1_unique_terms']))
    print("\nUnique to Doc2:", ', '.join(result['doc2_unique_terms']))
    
    # Example of batch comparison
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "The fast brown fox leaps over the sleepy dog",
        "A completely different sentence about cats",
    ]
    
    similarity_matrix = analyzer.batch_compare(documents)
    print("\nSimilarity Matrix:")
    print(similarity_matrix)
