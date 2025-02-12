from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from utils import check_compliance  # Import the updated check_compliance function
from models import db, Template  # Import db and Template from models
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Initialize the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/USER/Documents/Document_Compliance_System/instance/templates.db'  # Updated with absolute path
db.init_app(app)

# Redirect root URL to login page
@app.route('/')
def home():
    return redirect(url_for('login'))

# Sample user data for login (replace with your user management logic)
users = {
    "admin": generate_password_hash("password123")
}

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        template_file = request.files['template']
        student_file = request.files['student_file']

        if template_file and student_file:
            template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_file.filename)
            student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_file.filename)

            template_file.save(template_path)
            student_file.save(student_path)

            # Load the template content
            template_content = load_template_content()  # Function to load template content

            # Check compliance and get the score
            compliance_score, recommendations = check_compliance(template_content, student_path)

            # Process the results
            return render_template('result.html', score=compliance_score, recommendations=recommendations)

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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
