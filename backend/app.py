import json
import os
from flask import Flask, render_template, Response, redirect, session, url_for, jsonify, request
from PIL import Image
import io
import base64
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import cv2

app = Flask(__name__)

# --- CONFIGURACIÓN ---
GOOGLE_API_KEY = "AIzaSyAbOSJbQGO_PNC2pahShCYk1P6uDGvVhMw"
CAMERA_STREAM_URL = "http://192.168.1.35:8080/video"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Variables temporales ---
captured_image = None
result_image_data = None
result_points = None
result_analysis_text = None
is_processing = False

# --- MJPEG Stream Generator ---
def generate_stream():
    cap = cv2.VideoCapture(CAMERA_STREAM_URL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- Rutas ---
@app.route('/')
def index():
    return redirect('/chat')

@app.route('/registros')
def registros():
    return render_template('registros.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dash.html')

@app.route('/chat')
def chat_interface():
    return render_template(
        'chatbot.html',
        img_data=result_image_data,
        points_list=result_points if result_points else [],
        points_json=json.dumps(result_points if result_points else [])
    )


@app.route("/reset", methods=["POST"])
def reset():
    global captured_image, result_image_data, result_points, result_analysis_text, is_processing

    captured_image = None
    result_image_data = None
    result_points = None
    result_analysis_text = None
    is_processing = False

    return redirect(url_for("index"))

@app.route('/stream')
def stream():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capturar')
def capturar():
    global captured_image, result_image_data, result_points, result_analysis_text

    cap = cv2.VideoCapture(CAMERA_STREAM_URL)
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

    return redirect('/chat')

@app.route('/analizar', methods=['POST'])
def analizar():
    global captured_image, result_image_data, result_points, result_analysis_text

    if captured_image is None:
        return redirect(url_for('chat_interface'))

    # Redimensionar imagen para análisis
    image_resized = captured_image.resize(
        (800, int(800 * captured_image.size[1] / captured_image.size[0])),
        Image.Resampling.LANCZOS
    )

    # Llamada al modelo Gemini con la imagen y la instrucción
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
            temperature=0.2,  # Reducido para respuestas más estructuradas
            response_mime_type="application/json"  # IMPORTANTE: Solicita JSON directamente
        )
    )

    # Función mejorada para extraer JSON
    def parse_json(response_text):
        # Si ya es JSON válido, usarlo directamente
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
    try:
        result_points = parse_json(response.text)
        result_analysis_text = response.text
        print(f"Análisis exitoso. {len(result_points)} objetos detectados.")
    except Exception as e:
        print(f"Error procesando respuesta: {e}")
        result_points = [{"point": [500, 500], "label": "Error en análisis", "analisis": {}}]
        result_analysis_text = f"Error procesando la respuesta del modelo: {str(e)}"

    return redirect('/chat')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    global result_analysis_text, captured_image

    pregunta = request.form.get('pregunta')

    if not result_analysis_text or not captured_image:
        return jsonify({'respuesta': '❗ No se ha realizado ningún análisis de imagen aún.'})

    # Enviar imagen + análisis + pregunta
    image_resized = captured_image.resize(
        (800, int(800 * captured_image.size[1] / captured_image.size[0])),
        Image.Resampling.LANCZOS
    )

    response = model.generate_content(
        contents=[
            image_resized,
            f"""Este es el análisis previo realizado sobre la imagen:\n{result_analysis_text}
            Responde la siguiente pregunta basada únicamente en esta imagen:\n{pregunta}
            Responde claramente y en español."""
        ],
        generation_config=GenerationConfig(temperature=0.5)
    )

    return jsonify({'respuesta': response.text})

# --- Ejecutar ---
if __name__ == '__main__':
    app.run(debug=True)

