from backend import create_app, db
from backend.models import Usuario
from config import Config

app = create_app(Config)

def init_database():
    with app.app_context():
        # Crear todas las tablas
        db.create_all()
        
        # Crear usuario admin por defecto si no existe
        if not Usuario.query.filter_by(username='admin').first():
            admin = Usuario(
                username='admin',
                nombre_completo='Administrador del Sistema',
                rol='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Usuario admin creado: username=admin, password=admin123")
        
        print("Base de datos inicializada correctamente")

if __name__ == '__main__':
    init_database()