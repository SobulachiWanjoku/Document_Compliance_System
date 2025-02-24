import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from utils import check_compliance  # Import the updated check_compliance function
from werkzeug.utils import secure_filename # Import secure_filename for safe file handling
from models import db, Template # Import db and Template from models

app = Flask(__name__)
app.secret_key = 'SobulachiWanjoku'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Sample user data for login
users = {
    "admin": generate_password_hash("password123")
}

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
        if 'template' not in request.files or 'student_file' not in request.files:
            return {"error": "No file part in the request"}, 400
            return redirect(url_for('upload_file'))

        template_file = request.files['template']
        student_file = request.files['student_file']

        if template_file.filename == '' or student_file.filename == '':
            return {"error": "No selected file"}, 400
            return redirect(url_for('upload_file'))

        if not allowed_file(template_file.filename) or not allowed_file(student_file.filename):
            return {"error": "Invalid file type. Only PDF and DOCX files are allowed."}, 400
            return redirect(url_for('upload_file'))

        if template_file and student_file and allowed_file(student_file.filename):

            try:
                 # Sanitize file names
                template_filename = secure_filename(template_file.filename)
                student_filename = secure_filename(student_file.filename)

                # Save the template to the database
                new_template = Template(name=template_filename, content=template_file.read())
                db.session.add(new_template)
                db.session.commit()

                # Check if the student file is an image or DOCX
                student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename)
                student_file.save(student_path)

                # Ensure the file is processed correctly
                if not allowed_file(student_file.filename):
                    return {"error": "Invalid file type. Only PDF and DOCX files are allowed."}, 400


                # Retrieve the latest template for compliance checking
                template = Template.query.order_by(Template.id.desc()).first()

                # Check if template content is empty
                if not template.content:
                    flash('Template content is empty. Please upload a valid template.', 'error')
                    return redirect(url_for('upload_file'))

                try:
                    compliance_score, recommendations = check_compliance(template.content, student_path)
                except Exception as e:
                    return {"error": f"Compliance check failed: {str(e)}"}, 500

                print(f"Compliance Score: {compliance_score}, Recommendations: {recommendations}")  # Debug statement

                return render_template('result.html',score=compliance_score,recommendations=recommendations)

            except Exception as e:
                return {"error": f"An error occurred: {str(e)}"}, 500
                return redirect(url_for('upload_file'))

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
            return redirect(url_for('dashboard')) # Redirect to dashboard instead of upload_file
        
        flash("Invalid username or password")
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
        flash("Template deleted successfully.")
    else:
        flash("Template not found.")
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
