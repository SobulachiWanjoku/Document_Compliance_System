from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import pytesseract
from utils import check_compliance, generate_recommendations
from models import db, Template

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///templates.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()  # Create database tables

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        template_file = request.files['template']
        student_file = request.files['student_file']

        # Save the template to the database
        new_template = Template(name=template_file.filename, content=template_file.read())
        db.session.add(new_template)
        db.session.commit()

        # Save the student file temporarily
        student_path = os.path.join('uploads/', student_file.filename)
        student_file.save(student_path)

        # Retrieve the latest template for compliance checking
        template = Template.query.order_by(Template.id.desc()).first()
        compliance_score, recommendations = check_compliance(template.content, student_path)

        return render_template('result.html', score=compliance_score, recommendations=recommendations)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)