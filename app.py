import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils import check_compliance
from models import db, Template

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'SobulachiWanjoku'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
db.init_app(app)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Sample user data for login
users = {
    "admin": generate_password_hash("password123")
}

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
        # Check if both files are in the request
        if 'template' not in request.files or 'student_file' not in request.files:
            flash("Please select both files before uploading.", "error")
            return redirect(url_for('upload_file'))

        template_file = request.files['template']
        student_file = request.files['student_file']

        # Check if files are selected
        if template_file.filename == '' or student_file.filename == '':
            flash("No file selected. Please choose files to upload.", "error")
            return redirect(url_for('upload_file'))

        # Validate file extensions
        if not allowed_file(template_file.filename) or not allowed_file(student_file.filename):
            flash("Invalid file type. Only PDF and DOCX files are allowed.", "error")
            return redirect(url_for('upload_file'))

        try:
            # Sanitize file names
            template_filename = secure_filename(template_file.filename)
            student_filename = secure_filename(student_file.filename)

            # Save the template file to the upload folder
            template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
            template_file.save(template_path)

            # Retrieve the latest template for compliance checking
            template = Template.query.order_by(Template.id.desc()).first()

            # Check if template content is empty
            if not template.content:
                flash('Template content is empty. Please upload a valid template.', 'error')
                return redirect(url_for('upload_file'))

            # Save template path to the database
            new_template = Template(name=template_filename, path=template_path)
            db.session.add(new_template)
            db.session.commit()

            # Save student file to the upload folder
            student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename)
            student_file.save(student_path)

            # Retrieve the latest template for compliance checking
            template = Template.query.order_by(Template.id.desc()).first()

            # Perform compliance check
            try:
                compliance_score, recommendations = check_compliance(template.path, student_path)
                return render_template('result.html', score=compliance_score, recommendations=recommendations)
            except Exception as e:
                flash(f"Compliance check failed: {str(e)}", "error")
                return redirect(url_for('upload_file'))

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('upload_file'))

    return render_template('upload.html')

def load_template_content():
    # Implement logic to load the template content from the database or a file
    template = Template.query.order_by(Template.id.desc()).first()
    return template.content if template and template.content else "Your template content here."

@app.route('/gallery')
@login_required
def gallery():
    """Displays all uploaded templates for the logged-in user."""
    templates = Template.query.all()
    return render_template('gallery.html', templates=templates)

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
    if template:
        db.session.delete(template)
        db.session.commit()
        flash("Template deleted successfully.", "success")
    else:
        flash("Template not found.", "error")
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
    app.run(debug=True)
