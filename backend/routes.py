from flask import Blueprint, render_template, redirect, session, url_for, jsonify, request, Response, current_app
from flask_login import login_required, current_user, login_user, logout_user
from backend import db, login_manager
from backend.models import Usuario, Analisis, Evidencia
from PIL import Image
import io
import base64
import json
import cv2
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

bp = Blueprint('main', __name__)

# User loader para Flask-Login (aquí no hay circularidad)
@login_manager.user_loader
def load_user(user_id):
    return Usuario.query.get(int(user_id))

# Configuración Gemini (se inicializará en cada request que lo necesite)
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

@bp.route('/api/estadisticas')
@login_required
def api_estadisticas():
    # Obtener estadísticas actualizadas
    total_analisis = Analisis.query.count()
    total_evidencias = Evidencia.query.count()
    
    # Obtener datos para gráficos (misma lógica que en dashboard)
    siete_dias_atras = datetime.now() - timedelta(days=7)
    
    # Crear un diccionario con todos los días de la semana inicializados en 0
    actividad_dict = {}
    for i in range(7):
        fecha = siete_dias_atras + timedelta(days=i)
        fecha_str = fecha.strftime('%Y-%m-%d')
        actividad_dict[fecha_str] = 0
    
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
    
    # Preparar datos finales
    actividad_labels = []
    actividad_data = []
    
    for i in range(7):
        fecha = siete_dias_atras + timedelta(days=i)
        fecha_str = fecha.strftime('%Y-%m-%d')
        dia_nombre = fecha.strftime('%a')
        
        actividad_labels.append(f"{dia_nombre} {fecha.strftime('%d')}")
        actividad_data.append(actividad_dict.get(fecha_str, 0))
    
    # Distribución real
    total_con_evidencias = db.session.query(Analisis.id).join(Evidencia).distinct().count()
    total_sin_evidencias = total_analisis - total_con_evidencias
    
    if total_analisis == 0:
        distribucion_labels = ['Sin datos']
        distribucion_data = [1]
    else:
        distribucion_labels = ['Con evidencias', 'Sin evidencias']
        distribucion_data = [total_con_evidencias, total_sin_evidencias]
        
        if sum(distribucion_data) == 0:
            distribucion_labels = ['Sin datos']
            distribucion_data = [1]
    
    return jsonify({
        'total_analisis': total_analisis,
        'total_evidencias': total_evidencias,
        'actividad_labels': actividad_labels,
        'actividad_data': actividad_data,
        'distribucion_labels': distribucion_labels,
        'distribucion_data': distribucion_data
    })

@bp.route('/registros')
@login_required
def registros():
    # Filtrar por usuario logueado
    analisis_list = Analisis.query.filter_by(usuario_id=current_user.id) \
                                  .order_by(Analisis.fecha_analisis.desc()) \
                                  .all()

    # Convertir imágenes a base64
    for analisis in analisis_list:
        if analisis.imagen_analizada:
            analisis.imagen_analizada_b64 = base64.b64encode(
                analisis.imagen_analizada
            ).decode('utf-8')
        else:
            analisis.imagen_analizada_b64 = None

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
    camera_ip = request.form.get('camera_ip')
    if camera_ip:
        # Guardar en la sesión de este usuario
        session['camera_ip'] = camera_ip.strip()
    # Siempre volvemos al dashboard
    return redirect(url_for('main.dashboard'))


def get_user_camera_url():
    ip = session.get('camera_ip')
    if ip:
        return f"http://{ip}:8080/video"
    # fallback por defecto
    return "http://192.168.1.34:8080/video"


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

# MJPEG Stream Generator
# --- Generador de stream ---
def generate_stream(camera_url):
    cap = cv2.VideoCapture(camera_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


# --- Ruta de streaming ---
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

    # Codificamos la imagen capturada a base64
    buffered = io.BytesIO()
    captured_image.save(buffered, format="PNG")
    result_image_data = base64.b64encode(buffered.getvalue()).decode()

    # Reiniciamos análisis previo
    result_points = None
    result_analysis_text = None

    return redirect(url_for('main.chat_interface'))

@bp.route('/analizar', methods=['POST'])
@login_required
def analizar():
    global captured_image, result_image_data, result_points, result_analysis_text

    if captured_image is None:
        return redirect(url_for('main.chat_interface'))

    # Redimensionar imagen para análisis
    image_resized = captured_image.resize(
        (800, int(800 * captured_image.size[1] / captured_image.size[0])),
        Image.Resampling.LANCZOS
    )

    try:
        # Llamada al modelo Gemini
        model = get_gemini_model()
        response = model.generate_content(
            contents=[
                image_resized,
                """
                Eres un perito forense analizando una escena del crimen. Para cada objeto visible en la imagen:

                1. Identifica todos los objetos y su ubicación aproximada (coordenadas [y, x] normalizadas 0-1000)
                2. Para CADA objeto, genera un análisis detallado con el siguiente formato:

                {
                  "point": [y, x],  // coordenadas normalizadas 0-1000 (MANTENER ESTE FORMATO)
                  "label": "Nombre del objeto",
                  "analisis": {
                    "ubicacion": "Descripción de ubicación y posición",
                    "naturaleza": "Tipo de objeto y características",
                    "condicion": "Estado físico (intacto, roto, dañado, etc.)",
                    "indicios": "Huellas, fibras, manchas u otros indicios adheridos",
                    "pertinencia": "Relevancia para el caso investigado",
                    "valor_probatorio": "Valor como evidencia",
                    "observaciones": "Notas adicionales importantes"
                  }
                }

                Devuelve SOLAMENTE un array JSON válido con estos objetos, sin texto adicional.
                Los puntos deben estar en formato [y, x] y normalizados de 0 a 1000.
                Las etiquetas deben estar en español.
                """
            ],
            generation_config=GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        # Función para extraer JSON
        def parse_json(response_text):
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Intentar extraer JSON de bloques de código
                lines = response_text.splitlines()
                json_blocks = []
                in_json_block = False
                
                for line in lines:
                    if line.strip() in ["```json", "```"]:
                        in_json_block = not in_json_block
                        continue
                    if in_json_block:
                        json_blocks.append(line)
                
                if json_blocks:
                    try:
                        return json.loads("\n".join(json_blocks))
                    except json.JSONDecodeError:
                        pass
                
                # Último intento: buscar cualquier estructura que parezca JSON
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                
                # Fallback si todo falla
                return [{"point": [500, 500], "label": "Error en análisis", "analisis": {}}]

        # Codificar imagen para mostrarla luego
        buffered = io.BytesIO()
        image_resized.save(buffered, format="PNG")
        result_image_data = base64.b64encode(buffered.getvalue()).decode()

        # Guardar puntos con análisis completo
        result_points = parse_json(response.text)
        result_analysis_text = response.text
        print(f"Análisis exitoso. {len(result_points)} objetos detectados.")
            
    except Exception as e:
        print(f"Error procesando respuesta: {e}")
        result_points = [{"point": [500, 500], "label": "Error en análisis", "analisis": {}}]
        result_analysis_text = f"Error procesando la respuesta del modelo: {str(e)}"
    finally:
        # Redirigir de vuelta a la interfaz
        return redirect(url_for('main.chat_interface'))
    

@bp.route('/guardar_analisis', methods=['POST'])
@login_required
def guardar_analisis():
    global captured_image, result_points
    
    if not captured_image or not result_points:
        return jsonify({'error': 'No hay análisis para guardar'}), 400
    
    try:
        descripcion = request.form.get('descripcion', '')
        ubicacion = request.form.get('ubicacion', '')
        caso = request.form.get('caso', '')
        
        # Redimensionar imagen para el análisis
        image_resized = captured_image.resize(
            (800, int(800 * captured_image.size[1] / captured_image.size[0])),
            Image.Resampling.LANCZOS
        )
        
        # Convertir imágenes a bytes
        img_original_bytes = io.BytesIO()
        captured_image.save(img_original_bytes, format='PNG')
        
        img_analizada_bytes = io.BytesIO()
        image_resized.save(img_analizada_bytes, format='PNG')
        
        # Crear nuevo análisis en la base de datos
        nuevo_analisis = Analisis(
            usuario_id=current_user.id,
            imagen_original=img_original_bytes.getvalue(),
            imagen_analizada=img_analizada_bytes.getvalue(),
            descripcion=descripcion,
            ubicacion_escena=ubicacion,
            caso_asociado=caso
        )
        
        db.session.add(nuevo_analisis)
        db.session.flush()  # Para obtener el ID del análisis
        
        # Crear evidencias
        for evidencia_data in result_points:
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
                valor_probatorio=evidencia_data.get('analisis', {}).get('valor_probatorio', ''),
                observaciones=evidencia_data.get('analisis', {}).get('observaciones', '')
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