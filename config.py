import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-peridot-2024'
    
    # PostgreSQL Database - Usando tus credenciales
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://postgres:1234@localhost/peridotdb'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Gemini API
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyAbOSJbQGO_PNC2pahShCYk1P6uDGvVhMw')
    CAMERA_STREAM_URL = os.environ.get('CAMERA_STREAM_URL', 'http://192.168.1.35:8080/video')