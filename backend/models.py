from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Usar la misma instancia de db que se crea en __init__.py
from backend import db

class Usuario(db.Model, UserMixin):
    __tablename__ = 'usuarios'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    nombre_completo = db.Column(db.String(200), nullable=False)
    rol = db.Column(db.String(50), default='forense')
    fecha_creacion = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relación con análisis
    analisis = db.relationship('Analisis', backref='usuario', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<Usuario {self.username}>'

class Analisis(db.Model):
    __tablename__ = 'analisis'
    
    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuarios.id'), nullable=False)
    imagen_original = db.Column(db.LargeBinary, nullable=False)
    imagen_analizada = db.Column(db.LargeBinary)
    fecha_analisis = db.Column(db.DateTime, default=datetime.utcnow)
    descripcion = db.Column(db.Text)
    ubicacion_escena = db.Column(db.String(200))
    caso_asociado = db.Column(db.String(100))
    
    # Relación con evidencias
    evidencias = db.relationship('Evidencia', backref='analisis', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Analisis {self.id} - {self.caso_asociado}>'

class Evidencia(db.Model):
    __tablename__ = 'evidencias'
    
    id = db.Column(db.Integer, primary_key=True)
    analisis_id = db.Column(db.Integer, db.ForeignKey('analisis.id'), nullable=False)
    label = db.Column(db.String(200), nullable=False)
    point_x = db.Column(db.Integer, nullable=False)
    point_y = db.Column(db.Integer, nullable=False)
    ubicacion = db.Column(db.Text)
    naturaleza = db.Column(db.Text)
    condicion = db.Column(db.Text)
    indicios = db.Column(db.Text)
    pertinencia = db.Column(db.Text)
    valor_probatorio = db.Column(db.Text)
    observaciones = db.Column(db.Text)
    verificada = db.Column(db.Boolean, default=False)
    fecha_verificacion = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<Evidencia {self.id} - {self.label}>'