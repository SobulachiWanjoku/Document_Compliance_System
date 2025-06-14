import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from train_model import DocumentComplianceAnalyzer
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils import check_compliance, load_vectorizer, extract_text_from_file
from docx import Document  # Importing Document class for handling DOCX files
from models import db, Template
from models import User as UserModel
from flask_migrate import Migrate
from datetime import timedelta
import re
from markupsafe import escape
# Initialize Flask app


app = Flask(__name__)
app.secret_key = 'SobulachiWanjoku'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

# Session expiration and cookie security settings
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)  # Session expires after 10 minutes of inactivity
app.config['SESSION_COOKIE_SECURE'] = True  # Cookie sent only over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Cookie not accessible via JavaScript
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Helps prevent CSRF

# Configure logging
if not os.path.exists('logs'):
    os.mkdir('logs')
log_handler = RotatingFileHandler('logs/error.log', maxBytes=10240, backupCount=10)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Application startup')

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
    template_id = request.args.get('template_id', type=int)
    preloaded_template = None

    if template_id:
        preloaded_template = Template.query.get(template_id)
        if not preloaded_template or not os.path.exists(preloaded_template.path):
            flash("Selected template not found or file missing.", "error")
            return redirect(url_for('gallery'))

    if request.method == 'POST':
        preloaded_template_id = request.form.get('preloaded_template_id', type=int)
        template_file = request.files.get('template')
        student_file = request.files.get('student_file')

        if not student_file:
            flash("Please select the student file before uploading.", "error")
            return redirect(url_for('upload_file'))

        if preloaded_template_id:
            # Use preloaded template from database
            template = Template.query.get(preloaded_template_id)
            if not template or not os.path.exists(template.path):
                flash("Selected template not found or file missing.", "error")
                return redirect(url_for('upload_file'))
        else:
            # Require template file upload
            if not template_file:
                flash("Please select the template file before uploading.", "error")
                return redirect(url_for('upload_file'))

            if not allowed_file(template_file.filename):
                flash("Invalid template file type. Only PDF and DOCX files are allowed.", "error")
                return redirect(url_for('upload_file'))

        if not allowed_file(student_file.filename):
            flash("Invalid student file type. Only PDF and DOCX files are allowed.", "error")
            return redirect(url_for('upload_file'))

        try:
            # Check file size limits
            if not preloaded_template_id and template_file.content_length > 16 * 1024 * 1024:  # 16 MB limit
                flash("Template file is too large. Maximum size is 16 MB.", "error")
                return redirect(url_for('upload_file'))
            if student_file.content_length > 16 * 1024 * 1024:  # 16 MB limit
                flash("Student file is too large. Maximum size is 16 MB.", "error")
                return redirect(url_for('upload_file'))

            # Sanitize filenames
            if preloaded_template_id:
                template_filename = None
                template_path = None
            else:
                template_filename = secure_filename(template_file.filename)
                template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
                template_file.save(template_path)
                app.logger.info(f"Saving template path: {template_path}")  # Log the template path being saved

            flash("Template file uploaded successfully.", "success")

            # Save template in database (check for duplicates first)
            if not preloaded_template_id:
                existing_template = Template.query.filter(
                    (Template.name == template_filename) | 
                    (Template.path == template_path)
                ).first()
                
                if existing_template:
                    flash("This template already exists in the system.", "warning")
                else:
                    new_template = Template(name=template_filename, path=template_path)
                    db.session.add(new_template)
                    db.session.commit()
                    flash("Template file uploaded successfully.", "success")

            # Save student file
            student_filename = secure_filename(student_file.filename)
            student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename)
            student_file.save(student_path)

            # Retrieve template
            if preloaded_template_id:
                template = Template.query.get(preloaded_template_id)
                if not template or not os.path.exists(template.path):
                    flash("Selected template not found or file missing.", "error")
                    return redirect(url_for('upload_file'))
            else:
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

            student_formatting = analyzer.extract_formatting_from_docx(student_path)
            template_formatting = analyzer.extract_formatting_from_docx(template.path)
            if not template_text:
                app.logger.error("Template text is empty. Cannot evaluate compliance.")
                flash("Template text is empty. Please upload a valid template file.", "error")
                return redirect(url_for('upload_file'))

            if not student_text:
                app.logger.error("Student text is empty. Cannot evaluate compliance.")
                flash("Student file text extraction failed. Please upload a valid student file.", "error")
                return redirect(url_for('upload_file'))

            compliance_result = analyzer.evaluate_compliance(student_text, template_text, 
                student_formatting, 
                template_formatting, 
                analyzer.extract_headings_from_docx(student_path), 
                analyzer.extract_headings_from_docx(template.path))

            if not compliance_result:
                app.logger.error(f"Compliance evaluation returned empty result: {compliance_result}")
                flash("Failed to evaluate compliance. Please check the uploaded documents.", "error")
                return redirect(url_for('upload_file'))

            try:
                compliance_score = compliance_result['final_compliance_score']
                recommendations = compliance_result.get('recommendations', [])
            except KeyError as e:
                app.logger.error(f"KeyError accessing compliance_result: {e}, compliance_result content: {compliance_result}")
                flash("An error occurred during compliance evaluation. Please try again.", "error")
                return redirect(url_for('upload_file'))

            # Map alignment integer keys to human-readable strings
            alignment_map = {
                0: 'Left',
                1: 'Center',
                2: 'Right',
                3: 'Justify'
            }

            def map_alignments(formatting_dict):
                if 'alignments' in formatting_dict:
                    mapped = {}
                    for key, value in formatting_dict['alignments'].items():
                        try:
                            int_key = int(key)
                            mapped[alignment_map.get(int_key, f'Unknown({key})')] = value
                        except Exception:
                            mapped[key] = value
                    formatting_dict['alignments'] = mapped
                return formatting_dict

            template_formatting = map_alignments(template_formatting)
            student_formatting = map_alignments(student_formatting)

            # Pass formatting differences as dictionaries for better rendering
            formatting_differences = {
                'original': template_formatting,
                'formatted': student_formatting
            }

            return render_template('result.html', compliance_result={
                'final_compliance_score': compliance_score,
                'recommendations': recommendations,
                'formatting_differences': formatting_differences
            })


        except Exception as e:
            app.logger.error(f"User: {current_user.id} - An error occurred during file upload: {str(e)} - Template file: {template_file.filename if template_file else 'N/A'}, Student file: {student_file.filename if student_file else 'N/A'}, Request: {request.method} {request.path}")

            flash("An unexpected error occurred. Please try again later.", "error")
            return redirect(url_for('upload_file'))

    return render_template('upload.html', preloaded_template=preloaded_template)

@app.route('/gallery')
@login_required
def gallery():
    """Displays all uploaded templates for the logged-in user."""
    try:
        templates = [template for template in Template.query.all() if os.path.exists(template.path)]
        return render_template('gallery.html', templates=templates)
    except Exception as e:
        app.logger.error(f"An error occurred while fetching templates: {str(e)}")
        flash("An error occurred while fetching templates. Please try again later.", "error")
        return redirect(url_for('dashboard'))

@app.route('/reuse_template/<int:template_id>')
@login_required
def reuse_template(template_id):
    """Handle reuse of a selected template by redirecting to upload with template preloaded."""
    template = Template.query.get(template_id)
    if not template or not os.path.exists(template.path):
        flash("Template not found or file missing.", "error")
        return redirect(url_for('gallery'))
    # Redirect to upload page with template_id as query parameter
    return redirect(url_for('upload_file', template_id=template_id))

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    return UserModel.query.get(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Input validation
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template('login.html', title="Login")

        # Sanitize inputs
        username = escape(username)

        user = UserModel.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            # Mark session as permanent to enable session timeout
            session.permanent = True
            return redirect(url_for('dashboard'))

        flash("Invalid username or password", "error")
    return render_template('login.html', title="Login")

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Input validation
        if not username or not email or not password or not confirm_password:
            flash('Please fill out all fields.', 'error')
            return render_template('register.html')

        # Validate email format
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_regex, email):
            flash('Invalid email format.', 'error')
            return render_template('register.html')

        # Validate password strength (min 8 chars, at least one number and one letter)
        password_regex = r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$'
        if not re.match(password_regex, password):
            flash('Password must be at least 8 characters long and contain both letters and numbers.', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        # Check if username or email already exists
        existing_user = UserModel.query.filter((UserModel.username == username) | (UserModel.email == email)).first()
        if existing_user:
            flash('Username or email already exists.', 'error')
            return render_template('register.html')

        # Create new user
        new_user = UserModel(username=escape(username), email=escape(email))
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/delete_template/<int:template_id>', methods=['POST'])
@login_required
def delete_template(template_id):
    # Validate template_id
    if not isinstance(template_id, int) or template_id <= 0:
        flash("Invalid template ID.", "error")
        return redirect(url_for('gallery'))

    template = Template.query.get(template_id)
    
    if not template:
        flash("Template not found.", "error")
        return redirect(url_for('gallery'))
    
    app.logger.info(f"Attempting to delete template at path: {template.path}")

    if template.path:
        try:
            if os.path.exists(template.path):
                os.remove(template.path)  # Delete file from storage
            else:
                app.logger.warning(f"File not found: {template.path}")
        except Exception as e:
            app.logger.error(f"Error deleting file: {str(e)}")
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
        app.logger.error(f"An error occurred during logout: {str(e)}")
        flash('An error occurred during logout', 'error')
        return redirect(url_for('dashboard'))

# Centralized error handler to avoid exposing sensitive info
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}, Path: {request.path}, Method: {request.method}")
    return render_template('error.html', message="An unexpected error occurred. Please try again later."), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error(f"Unhandled Exception: {e}, Path: {request.path}, Method: {request.method}")
    return render_template('error.html', message="An unexpected error occurred. Please try again later."), 500

@app.errorhandler(404)
def not_found_error(error):
    if request.path == '/favicon.ico':
        app.logger.warning(f"Favicon not found: {request.path}")
        return '', 204  # No Content response to suppress error
    return render_template('error.html', message="Page not found."), 404

if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)
