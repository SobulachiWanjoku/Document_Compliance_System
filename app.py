import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from train_model import DocumentComplianceAnalyzer
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils import check_compliance, load_vectorizer, extract_text_from_file
from docx import Document  # Importing Document class for handling DOCX files
from models import db, Template
from flask_migrate import Migrate

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'SobulachiWanjoku'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

# Initialize SQLAlchemy and Migrate
db.init_app(app)
migrate = Migrate(app, db)  # Migrate is initialized here (after db.init_app)
# Instantiate the DocumentComplianceAnalyzer
analyzer = DocumentComplianceAnalyzer()

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Sample user data (Use a database in production)
users = {"admin": generate_password_hash("password123")}

# Helper function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Handle file upload and compliance checking"""
    if request.method == 'POST':
        template_file = request.files.get('template')
        student_file = request.files.get('student_file')

        if not template_file or not student_file:
            flash("Please select both files before uploading.", "error")
            return redirect(url_for('upload_file'))

        if not allowed_file(template_file.filename):
            flash("Invalid template file type. Only PDF and DOCX files are allowed.", "error")
            return redirect(url_for('upload_file'))
        if not allowed_file(student_file.filename):
            flash("Invalid student file type. Only PDF and DOCX files are allowed.", "error")
            return redirect(url_for('upload_file'))

            flash("Invalid file type. Only PDF and DOCX files are allowed.", "error")
            return redirect(url_for('upload_file'))

        try:
            # Check file size limits
            if template_file.content_length > 16 * 1024 * 1024:  # 16 MB limit
                flash("Template file is too large. Maximum size is 16 MB.", "error")
                return redirect(url_for('upload_file'))
            if student_file.content_length > 16 * 1024 * 1024:  # 16 MB limit
                flash("Student file is too large. Maximum size is 16 MB.", "error")
                return redirect(url_for('upload_file'))

            # Save template file

            template_filename = secure_filename(template_file.filename)
            template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
            template_file.save(template_path)

            logging.info(f"Saving template path: {template_path}")  # Log the template path being saved
            flash("Template file uploaded successfully.", "success")

            # Save template in database
            new_template = Template(name=template_filename, path=template_path)  # Ensure 'path' column exists
            db.session.add(new_template)
            db.session.commit()

            # Save student file
            student_filename = secure_filename(student_file.filename)
            student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename)
            student_file.save(student_path)

            # Retrieve latest template
            template = Template.query.order_by(Template.id.desc()).first()
            if not template:
                flash("No templates found in the database.", "error")
                return redirect(url_for('upload_file'))

            # Read template text
            if template.path.endswith('.docx'):
                doc = Document(template.path)
                template_text = "\n".join([para.text for para in doc.paragraphs])
            else:
                with open(template.path, "r", encoding="utf-8") as f:
                    template_text = f.read()

            # Load the fitted vectorizer using DocumentComplianceAnalyzer
            analyzer.load_model("vectorizer.pkl")
            vectorizer = analyzer.vectorizer

            # Read student document text
            student_text = extract_text_from_file(student_path)  # Ensure student_text is initialized
            if student_text is None:
                flash("Failed to extract text from the student file.", "error")
                return redirect(url_for('upload_file'))

            # Compliance check
            compliance_result = analyzer.evaluate_compliance(student_text, template_text, 
                analyzer.extract_formatting_from_docx(student_path), 
                analyzer.extract_formatting_from_docx(template_path), 
                analyzer.extract_headings_from_docx(student_path), 
                analyzer.extract_headings_from_docx(template_path))
            compliance_score = compliance_result['final_compliance_score']

            return render_template('result.html', score=compliance_score, recommendations=recommendations)

        except Exception as e:
            logging.error(f"An error occurred during file upload: {str(e)} - Template file: {template_file.filename}, Student file: {student_file.filename}")

            flash("An unexpected error occurred. Please try again later.", "error")
            return redirect(url_for('upload_file'))

    return render_template('upload.html')

@app.route('/gallery')
@login_required
def gallery():
    """Displays all uploaded templates for the logged-in user."""
    try:
        templates = [template for template in Template.query.all() if os.path.exists(template.path)]
        return render_template('gallery.html', templates=templates)
    except Exception as e:
        logging.error(f"An error occurred while fetching templates: {str(e)}")
        flash("An error occurred while fetching templates. Please try again later.", "error")
        return redirect(url_for('dashboard'))

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users else None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash("Invalid username or password", "error")
    return render_template('login.html', title="Login")

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/delete_template/<int:template_id>', methods=['POST'])
@login_required
def delete_template(template_id):
    template = Template.query.get(template_id)
    
    if not template:
        flash("Template not found.", "error")
        return redirect(url_for('gallery'))
    
    logging.info(f"Attempting to delete template at path: {template.path}")

    if template.path:
        try:
            if os.path.exists(template.path):
                os.remove(template.path)  # Delete file from storage
            else:
                logging.warning(f"File not found: {template.path}")
        except Exception as e:
            logging.error(f"Error deleting file: {str(e)}")
            flash("An error occurred while deleting the template file.", "error")
            return redirect(url_for('gallery'))
    
    db.session.delete(template)
    db.session.commit()
    flash("Template deleted successfully.", "success")
    
    return redirect(url_for('gallery'))

@app.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    try:
        logout_user()
        flash('You have been successfully logged out', 'success')
        return redirect(url_for('login'))
    except Exception as e:
        flash('An error occurred during logout', 'error')
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)
