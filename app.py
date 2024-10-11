from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
from fer import Video, FER
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Desactiva mensajes de depuración de TensorFlow relacionados con oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Inicializa la aplicación Flask
app = Flask(__name__)
CORS(app)

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://sysvita:j6tkmExB0Xes9z30gNiU6Ie39yEaDRGd@dpg-cs3bc8pu0jms73987300-a.oregon-postgres.render.com/g1_sysvita'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo de base de datos
class Resultados(db.Model):
    __tablename__ = 'resultados'
    id_resultado = db.Column(db.Integer, primary_key=True)
    enviado_tiempo = db.Column(db.DateTime, default=datetime.utcnow)
    angry = db.Column(db.Numeric, nullable=False)
    disgust = db.Column(db.Numeric, nullable=False)
    fear = db.Column(db.Numeric, nullable=False)
    happy = db.Column(db.Numeric, nullable=False)
    sad = db.Column(db.Numeric, nullable=False)
    surprise = db.Column(db.Numeric, nullable=False)
    neutral = db.Column(db.Numeric, nullable=False)
    emocion = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f"<Resultado {self.id_resultado} - {self.emocion}>"

# Verifica si la GPU está disponible
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print("Advertencia: No se encontró una GPU, el código se ejecutará en la CPU.")
else:
    print('Found GPU at: {}'.format(device_name))
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        # Guarda el archivo de video temporalmente
        video_file = request.files['video']
        videopath = os.path.join("uploads", video_file.filename)
        video_file.save(videopath)

        # Detección de emociones en el video
        detector = FER(mtcnn=True)
        video = Video(videopath)
        raw_data = video.analyze(detector, display=False)

        # Convertir a DataFrame de pandas
        df = video.to_pandas(raw_data)
        df = video.get_first_face(df)
        df = video.get_emotions(df)

        # Guardar los datos por frame en un archivo CSV
        csv_filename = 'emotions_per_frame.csv'
        df.to_csv(csv_filename, index=False)

        # Calcular el promedio de cada emoción
        emotions_avg = df.mean()
        emotions_avg_df = pd.DataFrame(emotions_avg, columns=["Average"]).transpose()
        emotions_avg_filename = 'emotions_average.csv'
        emotions_avg_df.to_csv(emotions_avg_filename, index=False)

        # Determinar el estado de ánimo final
        predominant_emotion = emotions_avg.idxmax()
        predominant_value = emotions_avg.max()

        # Generar gráfico de emociones por frame
        fig = df.plot(figsize=(20, 16), fontsize=26).get_figure()
        fig_filename = 'emotions_plot.png'
        fig.savefig(fig_filename)

        # Elimina el archivo de video temporal
        os.remove(videopath)

        # Guardar resultados en la base de datos
        resultado = Resultados(
            angry=emotions_avg['angry'],
            disgust=emotions_avg['disgust'],
            fear=emotions_avg['fear'],
            happy=emotions_avg['happy'],
            sad=emotions_avg['sad'],
            surprise=emotions_avg['surprise'],
            neutral=emotions_avg['neutral'],
            emocion=predominant_emotion
        )
        db.session.add(resultado)
        db.session.commit()

        # Devolver los resultados con el estado de ánimo predominante
        response = {
            "message": "Analysis completed successfully",
            "predominant_emotion": predominant_emotion,
            "average_value": predominant_value
        }
        return jsonify(response), 200

    except Exception as e:
        # En caso de un error, devuélvelo como respuesta
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    if filename == 'csv':
        return send_file('emotions_per_frame.csv', as_attachment=True)
    elif filename == 'avg_csv':
        return send_file('emotions_average.csv', as_attachment=True)
    elif filename == 'plot':
        return send_file('emotions_plot.png', as_attachment=True)
    else:
        return jsonify({"error": "Invalid filename"}), 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
