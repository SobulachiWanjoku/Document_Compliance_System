from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Template(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100), nullable=False)
    content = db.Column(db.LargeBinary, nullable=False)  # Store the file as binary
