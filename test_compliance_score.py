import unittest 
from train_model import DocumentComplianceAnalyzer

class TestFormattingCompliance(unittest.TestCase):
    def setUp(self):
        self.analyzer = DocumentComplianceAnalyzer()
        self.template_format = {
            "fonts": {"Arial", "Times New Roman"},
            "sizes": {12.0, 14.0},
            "bold_count": 10,
            "italic_count": 5,
            "underline_count": 2,
            "alignments": {0}  # Assuming 0 = left alignment
        }
        self.student_format_same = {
            "fonts": {"Arial", "Times New Roman"},
            "sizes": {12.0, 14.0},
            "bold_count": 10,
            "italic_count": 5,
            "underline_count": 2,
            "alignments": {0}
        }
        self.student_format_diff = {
            "fonts": {"Calibri"},  # Different font
            "sizes": {11.0},       # Different size
            "bold_count": 3,
            "italic_count": 1,
            "underline_count": 0,
            "alignments": {1}      # Assuming 1 = center alignment
        }

    def test_formatting_similarity_identical(self):
        score = self.analyzer.evaluate_compliance(
            student_doc="some text",
            template_doc="some text",  # Avoid identical bypass logic
            student_format=self.student_format_same,
            template_format=self.template_format,
            student_headings=["Heading 1"],
            template_headings=["Heading 1"]
        )["formatting_compliance"]
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_formatting_similarity_different(self):
        score = self.analyzer.evaluate_compliance(
            student_doc="some text",
            template_doc="some different text",  # Keep similarity < 0.99
            student_format=self.student_format_diff,
            template_format=self.template_format,
            student_headings=["Heading 1"],
            template_headings=["Heading 1"]
        )["formatting_compliance"]
        print("Formatting compliance score (different):", score)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

if __name__ == "__main__":
    unittest.main()
