from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from utils import check_compliance, generate_recommendations
from models import db, Template

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

db.init_app(app)

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'template' not in request.files or 'student_file' not in request.files:
            return "No file part in the request", 400

        template_file = request.files['template']
        student_file = request.files['student_file']

        if template_file.filename == '' or student_file.filename == '':
            return "No selected file", 400

        if not allowed_file(template_file.filename) or not allowed_file(student_file.filename):
            return "Invalid file type. Only PDF and DOCX files are allowed.", 400

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
                return "Template content is empty. Please upload a valid template.", 400

            compliance_score, recommendations = check_compliance(template.content, student_path)

            return render_template('result.html', score=compliance_score, recommendations=recommendations)

        except Exception as e:
            return f"An error occurred: {str(e)}", 500

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
