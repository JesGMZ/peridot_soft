from flask import Blueprint, flash, render_template, redirect, session, url_for, jsonify, request, Response, current_app
from flask_login import login_required, current_user, login_user, logout_user
from backend import db, login_manager
from backend.models import Usuario, Analisis, Evidencia
from PIL import Image
import io
import base64
import json
import cv2
import subprocess
import ffmpeg
import time
import numpy as np
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Frame, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.lib import colors
from datetime import datetime, timedelta
from sqlalchemy import func, extract
from werkzeug.security import check_password_hash

bp = Blueprint('main', __name__)


@login_manager.user_loader
def load_user(user_id):
    return Usuario.query.get(int(user_id))


def get_gemini_model():
    genai.configure(api_key=current_app.config['GOOGLE_API_KEY'])
    return genai.GenerativeModel('gemini-2.5-flash')

# Variables temporales
captured_image = None
result_image_data = None
result_points = None
result_analysis_text = None

@bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.chat_interface'))
    else:
        return redirect(url_for('auth.login'))

@bp.route('/dashboard')
@login_required
def dashboard():
    # Obtener estadísticas para el dashboard
    total_analisis = Analisis.query.count()
    total_evidencias = Evidencia.query.count()
    ultimos_analisis = Analisis.query.order_by(Analisis.fecha_analisis.desc()).limit(5).all()
    
    # Obtener datos para gráficos de actividad (últimos 7 días)
    siete_dias_atras = datetime.now() - timedelta(days=7)
    
    # Crear un diccionario con todos los días de la semana inicializados en 0
    actividad_dict = {}
    for i in range(7):
        fecha = siete_dias_atras + timedelta(days=i)
        fecha_str = fecha.strftime('%Y-%m-%d')
        actividad_dict[fecha_str] = 0
    
    # Obtener análisis reales de los últimos 7 días
    analisis_por_dia = db.session.query(
        func.date(Analisis.fecha_analisis).label('fecha'),
        func.count(Analisis.id).label('cantidad')
    ).filter(Analisis.fecha_analisis >= siete_dias_atras)\
     .group_by(func.date(Analisis.fecha_analisis)).all()
    
    # Actualizar el diccionario con datos reales
    for analisis in analisis_por_dia:
        fecha_str = analisis.fecha.strftime('%Y-%m-%d')
        if fecha_str in actividad_dict:
            actividad_dict[fecha_str] = analisis.cantidad
    
    # Preparar labels y datos para el gráfico
    actividad_labels = []
    actividad_data = []
    
    for i in range(7):
        fecha = siete_dias_atras + timedelta(days=i)
        fecha_str = fecha.strftime('%Y-%m-%d')
        dia_nombre = fecha.strftime('%a')  # Lun, Mar, etc.
        
        actividad_labels.append(f"{dia_nombre} {fecha.strftime('%d')}")
        actividad_data.append(actividad_dict.get(fecha_str, 0))
    
    # Obtener distribución real de análisis por estado/tipo
    # Si no tienes un campo de estado, usaremos datos basados en evidencias encontradas
    total_con_evidencias = db.session.query(Analisis.id).join(Evidencia).distinct().count()
    total_sin_evidencias = total_analisis - total_con_evidencias
    
    # Si no hay datos reales, usar datos de ejemplo realistas
    if total_analisis == 0:
        distribucion_labels = ['Sin datos']
        distribucion_data = [1]
    else:
        distribucion_labels = ['Con evidencias', 'Sin evidencias']
        distribucion_data = [total_con_evidencias, total_sin_evidencias]
        
        # Si ambos son 0, mostrar al menos algo
        if sum(distribucion_data) == 0:
            distribucion_labels = ['Sin datos']
            distribucion_data = [1]
    
    # Convertir imágenes a base64 para mostrar en el dashboard
    for analisis in ultimos_analisis:
        if analisis.imagen_analizada:
            analisis.imagen_analizada_b64 = base64.b64encode(analisis.imagen_analizada).decode('utf-8')
    
    print(f"Dashboard data - Actividad: {actividad_data}, Distribución: {distribucion_data}")
    
    return render_template(
        'dash.html',
        total_analisis=total_analisis,
        total_evidencias=total_evidencias,
        ultimos_analisis=ultimos_analisis,
        actividad_labels=json.dumps(actividad_labels),
        actividad_data=json.dumps(actividad_data),
        distribucion_labels=json.dumps(distribucion_labels),
        distribucion_data=json.dumps(distribucion_data)
    )

@bp.route('/registros')
@login_required
def registros():
    analisis_list = Analisis.query.filter_by(usuario_id=current_user.id) \
                                  .order_by(Analisis.fecha_analisis.desc()) \
                                  .all()

    for analisis in analisis_list:
        if analisis.imagen_analizada:
            analisis.imagen_analizada_b64 = base64.b64encode(
                analisis.imagen_analizada
            ).decode('utf-8')
        else:
            analisis.imagen_analizada_b64 = None

        verificadas = 0
        omitidas = 0

        for e in analisis.evidencias:
            if e.fecha_documentacion and e.fecha_identificacion:
                delta = e.fecha_documentacion - e.fecha_identificacion
                e.tiempo_min = round(delta.total_seconds() / 60, 2)
            else:
                e.tiempo_min = None

            if e.verificada:
                verificadas += 1
            if e.omitida:
                omitidas += 1

        analisis.verificadas_count = verificadas
        analisis.omitidas_count = omitidas

    return render_template('registros.html', analisis_list=analisis_list)

@bp.route('/analisis/<int:analisis_id>/eliminar', methods=['POST'])
@login_required
def eliminar_analisis(analisis_id):
    analisis = Analisis.query.get_or_404(analisis_id)

    # Eliminar evidencias asociadas primero
    Evidencia.query.filter_by(analisis_id=analisis.id).delete()

    db.session.delete(analisis)
    db.session.commit()

    return redirect(url_for('main.registros'))

@bp.route('/configurar_camara', methods=['POST'])
@login_required
def configurar_camara():
    nueva_url = request.form.get("camera_url")

    if not nueva_url:
        flash("Debe ingresar una URL válida de la cámara.", "danger")
        return redirect(url_for("main.dashboard"))

    # Guardar en el usuario actual
    current_user.camara_ip = nueva_url
    db.session.commit()

    flash(f"URL de la cámara configurada: {nueva_url}", "success")
    return redirect(url_for("main.dashboard"))


def get_user_camera_url():
    if current_user.is_authenticated and current_user.camara_ip:
        return current_user.camara_ip  # Aquí ya tienes rtsp://... completo
    # fallback por defecto
    return "rtsp://admin:RCACOI@192.168.1.37:554/Streaming/Channels/101"

@bp.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    global captured_image, result_image_data, result_points, result_analysis_text
    
    # Verificar que se haya enviado un archivo
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró archivo de imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': 'Formato de archivo no válido. Use: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
    
    try:
        image_data = file.read()
        
        captured_image = Image.open(io.BytesIO(image_data))
        
        if captured_image.mode != 'RGB':
            captured_image = captured_image.convert('RGB')
        
        max_size = 1920
        if captured_image.size[0] > max_size or captured_image.size[1] > max_size:
            captured_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        captured_image.save(buffered, format="PNG")
        result_image_data = base64.b64encode(buffered.getvalue()).decode()
        
        result_points = None
        result_analysis_text = None
        
        return jsonify({
            'success': True,
            'message': 'Imagen subida correctamente'
        })
        
    except Exception as e:
        print(f"Error al procesar imagen: {e}")
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

@bp.route('/analisis/<int:analisis_id>')
@login_required
def ver_analisis(analisis_id):
    analisis = Analisis.query.get_or_404(analisis_id)
    # Convertir imagen a base64 para mostrar en HTML
    if analisis.imagen_analizada:
        analisis.imagen_analizada_b64 = base64.b64encode(analisis.imagen_analizada).decode('utf-8')
    return render_template('detalle_analisis.html', analisis=analisis)

@bp.route('/chat')
@login_required
def chat_interface():
    return render_template(
        'chatbot.html',
        img_data=result_image_data,
        points_list=result_points if result_points else [],
        points_json=json.dumps(result_points if result_points else [])
    )

@bp.route("/reset", methods=["POST"])
@login_required
def reset():
    global captured_image, result_image_data, result_points, result_analysis_text
    captured_image = None
    result_image_data = None
    result_points = None
    result_analysis_text = None
    return redirect(url_for('main.index'))

def generate_stream(camera_url):
    """
    Genera stream usando solo OpenCV con mejor manejo de errores
    """
    print(f"Conectando con OpenCV a: {camera_url}")
    
    # Variables de control
    MAX_RECONNECT_ATTEMPTS = 5
    reconnect_delay = 2
    last_frame_time = time.time()
    TIMEOUT_SECONDS = 30  # Timeout de 30 segundos
    
    while True:
        cap = None
        try:
            # Configurar captura de video con parámetros específicos para RTSP
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            
            # Configuraciones para RTSP
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
            # Configurar timeout (en milisegundos)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
            
            if not cap.isOpened():
                print(f"Error: No se pudo abrir la cámara: {camera_url}")
                error_frame = create_error_image("No se pudo conectar")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       error_frame + b'\r\n')
                time.sleep(reconnect_delay)
                continue
            
            print("Cámara conectada correctamente")
            reconnect_delay = 2  # Resetear delay después de conexión exitosa
            
            while True:
                try:
                    # Verificar timeout
                    current_time = time.time()
                    if current_time - last_frame_time > TIMEOUT_SECONDS:
                        print(f"Timeout: No se recibieron frames por {TIMEOUT_SECONDS} segundos")
                        break
                    
                    # Leer frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Error leyendo frame, reintentando...")
                        # Contador de frames fallidos consecutivos
                        time.sleep(0.1)
                        continue
                    
                    # Actualizar tiempo del último frame exitoso
                    last_frame_time = current_time
                    
                    # Redimensionar si es necesario para reducir carga
                    frame = cv2.resize(frame, (640, 360))
                    
                    # Codificar a JPEG con calidad media
                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
                    ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
                    
                    if ret:
                        frame_bytes = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               frame_bytes + b'\r\n')
                    else:
                        print("Error codificando imagen")
                    
                    # Controlar FPS
                    time.sleep(0.066)  # ~15 FPS para reducir carga
                    
                except Exception as e:
                    print(f"Error procesando frame: {e}")
                    break
                    
        except Exception as e:
            print(f"Error en stream principal: {e}")
            
        finally:
            # Liberar recursos
            if cap is not None:
                cap.release()
            
        # Esperar antes de reconectar
        print(f"Reconectando en {reconnect_delay} segundos...")
        time.sleep(reconnect_delay)
        
        # Aumentar delay exponencialmente hasta máximo
        reconnect_delay = min(reconnect_delay * 1.5, 10)

def create_error_image(message):
    """Crear una imagen de error simple"""

    # Crear imagen negra
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Añadir texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = message
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (360 + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
    
    # Codificar a JPEG
    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

@bp.route('/stream')
@login_required
def stream():
    camera_url = get_user_camera_url()
    return Response(generate_stream(camera_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Ruta de captura ---
@bp.route('/capturar')
@login_required
def capturar():
    global captured_image, result_image_data, result_points, result_analysis_text

    cap = cv2.VideoCapture(get_user_camera_url())
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Error al capturar imagen", 500

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(frame_rgb)

    buffered = io.BytesIO()
    captured_image.save(buffered, format="PNG")
    result_image_data = base64.b64encode(buffered.getvalue()).decode()

    result_points = None
    result_analysis_text = None

    return redirect(url_for('main.chat_interface'))

@bp.route('/analizar', methods=['POST'])
@login_required
def analizar():
    global captured_image, result_image_data, result_points, result_analysis_text

    if captured_image is None:
        return redirect(url_for('main.chat_interface'))

    image_resized = captured_image.resize(
        (800, int(800 * captured_image.size[1] / captured_image.size[0])),
        Image.Resampling.LANCZOS
    )

    try:
        # 🔹 Medimos tiempo total de análisis
        tiempo_inicio_general = datetime.utcnow()

        model = get_gemini_model()
        response = model.generate_content(
            contents=[
                image_resized,
                """
                Eres un perito forense analizando una escena del crimen. Para cada objeto visible en la imagen:

                1. Identifica todos los objetos y su ubicación aproximada (coordenadas [y, x] normalizadas 0-1000)
                2. Para CADA objeto, genera un análisis detallado con el siguiente formato:

                {
                  "point": [y, x],
                  "label": "Nombre del objeto",
                  "analisis": {
                    "ubicacion": "Descripción de ubicación y posición",
                    "naturaleza": "Tipo de objeto y características",
                    "condicion": "Estado físico (intacto, roto, dañado, etc.)",
                    "indicios": "Huellas, fibras, manchas u otros indicios adheridos",
                    "pertinencia": "Relevancia para el caso investigado",
                    "valor_probatorio": "Clasifica como 'alto', 'medio' o 'bajo'",
                    "observaciones": "Notas adicionales importantes"
                  }
                }

                Devuelve SOLAMENTE un array JSON válido.
                """
            ],
            generation_config=GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        tiempo_fin_general = datetime.utcnow()
        duracion_total = (tiempo_fin_general - tiempo_inicio_general).total_seconds()

        # 🔹 Función para extraer JSON
        def parse_json(response_text):
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                return json.loads(match.group()) if match else []

        buffered = io.BytesIO()
        image_resized.save(buffered, format="PNG")
        result_image_data = base64.b64encode(buffered.getvalue()).decode()

        result_points = parse_json(response.text)

        # 🔹 Agregar tiempos realistas por evidencia
        for p in result_points:
            p["fecha_identificacion"] = tiempo_fin_general.isoformat()
            p["tiempo_inicio"] = tiempo_inicio_general.isoformat()
            p["tiempo_fin"] = tiempo_fin_general.isoformat()
            p["duracion_segundos"] = round(duracion_total, 3)  # segundos con 3 decimales

        result_analysis_text = response.text

        print(f"✅ Análisis exitoso. {len(result_points)} objetos detectados en {duracion_total:.3f} segundos.")
        for i, p in enumerate(result_points, 1):
            print(f"   Evidencia {i}: {p['label']} (duración total: {p['duracion_segundos']} s)")

    except Exception as e:
        print(f"❌ Error procesando respuesta: {e}")
        result_points = [{"point": [500, 500], "label": "Error en análisis", "analisis": {}}]
        result_analysis_text = f"Error procesando la respuesta del modelo: {str(e)}"

    finally:
        return redirect(url_for('main.chat_interface'))


@bp.route('/guardar_analisis', methods=['POST'])
@login_required
def guardar_analisis():
    """
    Guarda el análisis y sus evidencias, registrando:
    - fecha_verificacion y usuario_verificador_id si la evidencia viene verificada.
    - es_clave=True automáticamente si valor_probatorio es 'alto'.
    - Calcula tiempo de inicio, fin y duración por evidencia.
    """
    global captured_image, result_points

    if not captured_image or not result_points:
        return jsonify({'error': 'No hay análisis para guardar'}), 400

    try:
        descripcion = request.form.get('descripcion', '')
        ubicacion = request.form.get('ubicacion', '')
        caso = request.form.get('caso', '')

        # Redimensionar imagen analizada
        image_resized = captured_image.resize(
            (800, int(800 * captured_image.size[1] / captured_image.size[0])),
            Image.Resampling.LANCZOS
        )

        # Convertir imágenes a bytes
        img_original_bytes = io.BytesIO()
        captured_image.save(img_original_bytes, format='PNG')
        img_analizada_bytes = io.BytesIO()
        image_resized.save(img_analizada_bytes, format='PNG')

        # Crear análisis principal
        nuevo_analisis = Analisis(
            usuario_id=current_user.id,
            imagen_original=img_original_bytes.getvalue(),
            imagen_analizada=img_analizada_bytes.getvalue(),
            descripcion=descripcion,
            ubicacion_escena=ubicacion,
            caso_asociado=caso
        )
        db.session.add(nuevo_analisis)
        db.session.flush()  # ✅ obtener ID antes del commit

        from dateutil.parser import parse

        for evidencia_data in result_points:
            # --- Fecha de identificación ---
            fecha_ident = evidencia_data.get("fecha_identificacion")
            if fecha_ident:
                try:
                    fecha_ident = parse(fecha_ident)
                except Exception:
                    fecha_ident = datetime.utcnow()
            else:
                fecha_ident = datetime.utcnow()

            # --- Tiempos por evidencia ---
            tiempo_inicio_str = evidencia_data.get("tiempo_inicio")
            tiempo_fin_str = evidencia_data.get("tiempo_fin")

            try:
                tiempo_inicio = parse(tiempo_inicio_str) if tiempo_inicio_str else fecha_ident
                tiempo_fin = parse(tiempo_fin_str) if tiempo_fin_str else datetime.utcnow()
                duracion_segundos = (tiempo_fin - tiempo_inicio).total_seconds()
            except Exception:
                tiempo_inicio = fecha_ident
                tiempo_fin = datetime.utcnow()
                duracion_segundos = (tiempo_fin - tiempo_inicio).total_seconds()

            # --- Estado de verificación ---
            verificada = evidencia_data.get("verificada", False)

            # --- Valor probatorio ---
            valor_probatorio = evidencia_data.get("analisis", {}).get("valor_probatorio", "").strip().lower()

            # --- Definición automática ---
            if valor_probatorio == "alto":
                es_clave = True
                omitida = False
            elif verificada:
                es_clave = True
                omitida = False
            else:
                es_clave = False
                omitida = True

            # --- Campos de verificación ---
            if verificada:
                fecha_verificacion = datetime.utcnow()
                usuario_verificador_id = current_user.id
            else:
                fecha_verificacion = None
                usuario_verificador_id = None

            # --- Creación de evidencia ---
            nueva_evidencia = Evidencia(
                analisis_id=nuevo_analisis.id,
                label=evidencia_data.get('label', ''),
                point_x=evidencia_data.get('point', [0, 0])[1],
                point_y=evidencia_data.get('point', [0, 0])[0],
                ubicacion=evidencia_data.get('analisis', {}).get('ubicacion', ''),
                naturaleza=evidencia_data.get('analisis', {}).get('naturaleza', ''),
                condicion=evidencia_data.get('analisis', {}).get('condicion', ''),
                indicios=evidencia_data.get('analisis', {}).get('indicios', ''),
                pertinencia=evidencia_data.get('analisis', {}).get('pertinencia', ''),
                valor_probatorio=valor_probatorio,
                observaciones=evidencia_data.get('analisis', {}).get('observaciones', ''),
                fecha_identificacion=fecha_ident,
                fecha_documentacion=datetime.utcnow(),
                tiempo_inicio=tiempo_inicio,
                tiempo_fin=tiempo_fin,
                duracion_segundos=duracion_segundos,
                es_clave=es_clave,
                omitida=omitida,
                verificada=verificada,
                fecha_verificacion=fecha_verificacion,
                usuario_verificador_id=usuario_verificador_id
            )
            db.session.add(nueva_evidencia)

        db.session.commit()

        return jsonify({
            'success': True,
            'analisis_id': nuevo_analisis.id,
            'message': 'Análisis guardado correctamente'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Error: {str(e)}'}), 500


@bp.route('/marcar_verificado', methods=['POST'])
@login_required
def marcar_verificado():
    data = request.get_json()
    index = data.get("index")
    estado = data.get("estado")
    try:
        index = int(index)
        if 0 <= index < len(result_points):
            result_points[index]["verificada"] = bool(estado)
            if estado:
                result_points[index]["es_clave"] = True
                result_points[index]["omitida"] = False
            else:
                result_points[index]["es_clave"] = False
                result_points[index]["omitida"] = True
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Índice fuera de rango"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/analisis/<int:analisis_id>/exportar')
@login_required
def exportar_pdf(analisis_id):
    analisis = Analisis.query.get_or_404(analisis_id)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']

    # Título
    elements.append(Paragraph(f"Reporte de Análisis - {analisis.caso_asociado or 'Sin código'}", title_style))
    elements.append(Spacer(1, 12))

    # Datos generales
    elements.append(Paragraph(f"<b>Fecha de análisis:</b> {analisis.fecha_analisis.strftime('%d/%m/%Y')}", normal_style))
    elements.append(Paragraph(f"<b>Ubicación:</b> {analisis.ubicacion_escena or 'No especificada'}", normal_style))
    elements.append(Paragraph(f"<b>Descripción:</b> {analisis.descripcion or 'Sin descripción'}", normal_style))
    elements.append(Spacer(1, 12))

    # Imagen (si existe)
    if analisis.imagen_analizada:
        img_data = io.BytesIO(analisis.imagen_analizada)
        try:
            img = RLImage(img_data, width=400, height=250)
            elements.append(img)
            elements.append(Spacer(1, 12))
        except:
            pass

    # Evidencias
    elements.append(Paragraph("Evidencias", subtitle_style))
    elements.append(Spacer(1, 6))

    if analisis.evidencias:
        for i, e in enumerate(analisis.evidencias, start=1):
            elements.append(Paragraph(f"Evidencia {i}", styles["Heading3"]))
            elements.append(Spacer(1, 4))

            # Usamos Paragraph para que los textos largos hagan word wrap
            data = [
                ["Etiqueta", Paragraph(e.label or "N/A", normal_style)],
                ["Ubicación", Paragraph(e.ubicacion or "N/A", normal_style)],
                ["Naturaleza", Paragraph(e.naturaleza or "N/A", normal_style)],
                ["Condición", Paragraph(e.condicion or "N/A", normal_style)],
                ["Indicios", Paragraph(e.indicios or "N/A", normal_style)],
                ["Pertinencia", Paragraph(e.pertinencia or "N/A", normal_style)],
                ["Valor probatorio", Paragraph(e.valor_probatorio or "N/A", normal_style)],
                ["Observaciones", Paragraph(e.observaciones or "N/A", normal_style)],
            ]

            table = Table(data, colWidths=[4*cm, 10*cm])
            table.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#5D6D7E")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AAB7B8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D6EAF8")),  # fondo solo para los nombres de campo
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1B2631")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 14))
    else:
        elements.append(Paragraph("No se registraron evidencias.", normal_style))

    # Generar PDF
    doc.build(elements)
    buffer.seek(0)

    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": f"attachment;filename=analisis_{analisis_id}.pdf"})

@bp.route('/analisis_usuarios')
@login_required
def analisis_usuarios():
    analisis_list = Analisis.query.filter(Analisis.usuario_id != current_user.id) \
                                  .order_by(Analisis.fecha_analisis.desc()) \
                                  .all()

    for analisis in analisis_list:
        if analisis.imagen_analizada:
            analisis.imagen_analizada_b64 = base64.b64encode(
                analisis.imagen_analizada
            ).decode('utf-8')
        else:
            analisis.imagen_analizada_b64 = None

    return render_template('informes.html', analisis_list=analisis_list)

@bp.route('/analisis/<int:analisis_id>/indicadores')
@login_required
def indicadores_analisis(analisis_id):
    evidencias = Evidencia.query.filter_by(analisis_id=analisis_id).all()

    if not evidencias:
        return jsonify({
            "promedio_identificacion": 0,
            "promedio_documentacion": 0,
            "verificadas": 0
        })

    # --- Cálculos ---
    total_tiempo_identificacion = 0
    total_tiempo_documentacion = 0
    count_identificacion = 0
    count_documentacion = 0
    verificadas = 0

    for e in evidencias:
        # Tiempo entre creación de Analisis y detección
        if e.fecha_identificacion:
            # Diferencia en segundos desde inicio del análisis
            inicio = e.analisis.fecha_analisis
            total_tiempo_identificacion += (e.fecha_identificacion - inicio).total_seconds()
            count_identificacion += 1

        # Tiempo entre identificación y documentación
        if e.fecha_documentacion and e.fecha_identificacion:
            total_tiempo_documentacion += (e.fecha_documentacion - e.fecha_identificacion).total_seconds()
            count_documentacion += 1

        if e.verificada:
            verificadas += 1

    promedio_identificacion = round(total_tiempo_identificacion / count_identificacion, 2) if count_identificacion else 0
    promedio_documentacion = round(total_tiempo_documentacion / count_documentacion, 2) if count_documentacion else 0

    return jsonify({
        "promedio_identificacion": promedio_identificacion,  # segundos
        "promedio_documentacion": promedio_documentacion,    # segundos
        "verificadas": verificadas
    })

@bp.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    global result_analysis_text, captured_image

    pregunta = request.form.get('pregunta')

    if not result_analysis_text or not captured_image:
        return jsonify({'respuesta': '❗ No se ha realizado ningún análisis de imagen aún.'})

    # Redimensionar imagen manteniendo relación de aspecto
    image_resized = captured_image.resize(
        (800, int(800 * captured_image.size[1] / captured_image.size[0])),
        Image.Resampling.LANCZOS
    )

    model = get_gemini_model()
    
    # Prompt más específico para evitar formato markdown
    prompt = f"""
    ANALISIS PREVIO:
    {result_analysis_text}
    
    PREGUNTA DEL USUARIO:
    {pregunta}
    
    INSTRUCCIONES:
    - Responde únicamente basado en la imagen y el análisis proporcionado
    - Usa un lenguaje claro y conciso en español
    - Evita usar formato markdown como **negritas** o ## encabezados
    - Proporciona una respuesta directa y bien estructurada
    - Si no hay información suficiente, indica que no se puede determinar
    """
    
    response = model.generate_content(
        contents=[image_resized, prompt],
        generation_config=GenerationConfig(
            temperature=0.3,  # Temperatura más baja para respuestas más precisas
            max_output_tokens=1000
        )
    )

    # Limpiar posible formato markdown de la respuesta
    respuesta_limpia = response.text.replace('**', '').replace('##', '').strip()
    
    return jsonify({'respuesta': respuesta_limpia})

@bp.route('/configuracion')
@login_required
def configuracion():
    return render_template('configuracion.html')

@bp.route('/editar_usuario', methods=['POST'])
@login_required
def editar_usuario():
    try:
        # Obtener datos del formulario
        nombre_completo = request.form.get('nombre_completo')
        username = request.form.get('username')
        camara_ip = request.form.get('camara_ip')
        password_actual = request.form.get('password_actual')
        nueva_password = request.form.get('nueva_password')
        confirmar_password = request.form.get('confirmar_password')
        
        # Verificar que el usuario existe
        usuario = Usuario.query.get(current_user.id)
        if not usuario:
            flash('Usuario no encontrado', 'error')
            return redirect(url_for('main.configuracion'))
        
        # Validar username único (si se cambió)
        if username != usuario.username:
            usuario_existente = Usuario.query.filter_by(username=username).first()
            if usuario_existente and usuario_existente.id != usuario.id:
                flash('El nombre de usuario ya está en uso', 'error')
                return redirect(url_for('main.configuracion'))
            usuario.username = username
        
        # Actualizar campos básicos
        usuario.nombre_completo = nombre_completo
        usuario.camara_ip = camara_ip
        
        # Cambiar contraseña si se proporcionó
        if password_actual and nueva_password and confirmar_password:
            if not usuario.check_password(password_actual):
                flash('La contraseña actual es incorrecta', 'error')
                return redirect(url_for('main.configuracion'))
            
            if nueva_password != confirmar_password:
                flash('Las nuevas contraseñas no coinciden', 'error')
                return redirect(url_for('main.configuracion'))
            
            if len(nueva_password) < 6:
                flash('La nueva contraseña debe tener al menos 6 caracteres', 'error')
                return redirect(url_for('main.configuracion'))
            
            usuario.set_password(nueva_password)
            flash('Contraseña actualizada correctamente', 'success')
        
        # Guardar cambios
        db.session.commit()
        flash('Perfil actualizado correctamente', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error al actualizar el perfil: {str(e)}', 'error')
    
    return redirect(url_for('main.configuracion'))

@bp.route('/obtener_datos_usuario')
@login_required
def obtener_datos_usuario():
    """Obtener datos del usuario para AJAX"""
    usuario = Usuario.query.get(current_user.id)
    if usuario:
        return jsonify({
            'id': usuario.id,
            'username': usuario.username,
            'nombre_completo': usuario.nombre_completo,
            'rol': usuario.rol,
            'fecha_creacion': usuario.fecha_creacion.strftime('%d/%m/%Y'),
            'camara_ip': usuario.camara_ip or '',
            'total_analisis': len(usuario.analisis),
            'total_evidencias': sum(len(analisis.evidencias) for analisis in usuario.analisis)
        })
    return jsonify({'error': 'Usuario no encontrado'}), 404