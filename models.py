from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Template(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    path = db.Column(db.String(255), nullable=False, unique=True)  # Store file path instead of binary content
