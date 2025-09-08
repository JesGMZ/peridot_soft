from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required
from backend import db
from backend.models import Usuario

bp = Blueprint('auth', __name__)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = Usuario.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.index'))
        else:
            flash('Usuario o contraseña incorrectos', 'danger')
    
    return render_template('login.html')

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        nombre_completo = request.form.get('nombre_completo')
        
        if Usuario.query.filter_by(username=username).first():
            flash('El usuario ya existe', 'danger')
            return redirect(url_for('auth.register'))
        
        user = Usuario(username=username, nombre_completo=nombre_completo)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Usuario registrado correctamente', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))