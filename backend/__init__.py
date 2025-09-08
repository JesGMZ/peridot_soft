from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config

# Inicializar extensiones primero
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Inicializar extensiones
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    
    # Registrar blueprints (importar aquí para evitar circularidad)
    with app.app_context():
        from backend.routes import bp as main_bp
        from backend.auth import bp as auth_bp
        
        app.register_blueprint(main_bp)
        app.register_blueprint(auth_bp, url_prefix='/auth')
    
    return app
