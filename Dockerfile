# Usa una imagen base de Python
FROM python:3.11-slim

# Instala las dependencias del sistema que necesita OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de tu proyecto al contenedor
COPY . .

# Crea el directorio de uploads
RUN mkdir -p uploads

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto en el que tu aplicación escuchará
EXPOSE 5000

# Comando para ejecutar tu aplicación
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
