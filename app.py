import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from utils import check_compliance  # Import the updated check_compliance function
from werkzeug.utils import secure_filename  # Import secure_filename for safe file handling

from models import db, Template  # Import db and Template from models


app = Flask(__name__)
app.secret_key = 'SobulachiWanjoku'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
db.init_app(app)  # Initialize the SQLAlchemy instance with the Flask app

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Redirect root URL to login page
@app.route('/')
def home():
    return redirect(url_for('login'))

# Sample user data for login (replace with your user management logic)
users = {
    "admin": generate_password_hash("password123")
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'template' not in request.files or 'student_file' not in request.files:
            return {"error": "No file part in the request"}, 400

        template_file = request.files['template']
        student_file = request.files['student_file']

        if template_file.filename == '' or student_file.filename == '':
            return {"error": "No selected file"}, 400

        if not allowed_file(template_file.filename) or not allowed_file(student_file.filename):
            return {"error": "Invalid file type. Only PDF and DOCX files are allowed."}, 400


        if template_file and student_file:
            try:
                # Sanitize file names
                template_filename = secure_filename(template_file.filename)
                student_filename = secure_filename(student_file.filename)

                # Save the template to the database
                new_template = Template(name=template_filename, content=template_file.read())
                db.session.add(new_template)
                db.session.commit()

                # Save the student file temporarily
                student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename)
                student_file.save(student_path)

                # Retrieve the latest template for compliance checking
                template = Template.query.order_by(Template.id.desc()).first()

                # Check if template content is empty
                if not template.content:
                    return {"error": "Template content is empty. Please upload a valid template."}, 400

                compliance_score, recommendations = check_compliance(template.content, student_path)

                return render_template('result.html', score=compliance_score, recommendations=recommendations)

            except Exception as e:
                return {"error": f"An error occurred: {str(e)}"}, 500

    return render_template('upload.html')

def load_template_content():
    # Implement logic to load the template content from the database or a file
    template = Template.query.order_by(Template.id.desc()).first()
    return template.content if template and template.content else "Your template content here."

@app.route('/gallery')
@login_required
def gallery():
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
            return redirect(url_for('dashboard'))  # Redirect to dashboard instead of upload_file
        
        flash("Invalid username or password")
    return render_template('login.html', title="Login")

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')  # Render the new dashboard page

@app.route('/delete_template/<int:template_id>', methods=['POST'])
@login_required
def delete_template(template_id):
    template = Template.query.get(template_id)
    if template:
        db.session.delete(template)
        db.session.commit()
        flash("Template deleted successfully.")
    else:
        flash("Template not found.")
    return redirect(url_for('gallery'))

def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
