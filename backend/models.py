from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Numeric
from backend import db

class Usuario(db.Model, UserMixin):
    __tablename__ = 'usuarios'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    nombre_completo = db.Column(db.String(200), nullable=False)
    rol = db.Column(db.String(50), default='forense')
    fecha_creacion = db.Column(db.DateTime, default=datetime.utcnow)
    camara_ip = db.Column(db.Text)
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

    # --- NUEVO: para métricas globales del análisis ---
    tiempo_total = db.Column(db.Interval)  # Duración total del análisis
    numero_evidencias = db.Column(db.Integer, default=0)  # Evidencias totales detectadas

    # Relación con evidencias
    evidencias = db.relationship(
        'Evidencia',
        backref='analisis',
        lazy=True,
        cascade='all, delete-orphan'
    )
    
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

    # --- NUEVOS CAMPOS PARA INDICADORES ---
    fecha_identificacion = db.Column(db.DateTime, default=datetime.utcnow)   # Cuándo fue detectada
    fecha_documentacion = db.Column(db.DateTime)                             # Cuándo terminó su documentación
    tiempo_inicio = db.Column(db.DateTime, nullable=True)
    tiempo_fin = db.Column(db.DateTime, nullable=True)
    duracion_segundos = db.Column(Numeric(10, 2), nullable=True)
    es_clave = db.Column(db.Boolean, default=False, index=True)               # Evidencia clave
    omitida = db.Column(db.Boolean, default=False, index=True)                 # Si fue omitida
    verificada = db.Column(db.Boolean, default=False, index=True)             # Validación por experto
    fecha_verificacion = db.Column(db.DateTime)
    usuario_verificador_id = db.Column(
        db.Integer,
        db.ForeignKey('usuarios.id'),
        index=True
    )  # Quién validó

    def __repr__(self):
        return f'<Evidencia {self.id} - {self.label}>'
