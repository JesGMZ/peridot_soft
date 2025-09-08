import os
from dotenv import load_dotenv

# Cargar variables del archivo .env
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    # Clave secreta para Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-peridot-2024'
    
    # PostgreSQL (Supabase)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Gemini API
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyAbOSJbQGO_PNC2pahShCYk1P6uDGvVhMw')

    # Cámara (por defecto la IP local, pero configurable)
    CAMERA_STREAM_URL = os.environ.get('CAMERA_STREAM_URL', 'http://192.168.1.34:8080/video')
