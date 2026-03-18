# TechView

Sistema de asistencia basado en visión por computadora que detecta objetos en tiempo real y genera alertas por voz. Diseñado con arquitectura cliente-servidor para entornos como Raspberry Pi y Windows.

## 🚀 Características

- Detección de objetos en tiempo real (YOLOv8)
- Identificación de personas, vehículos y objetos de entorno
- Sistema de alertas por voz y sonidos
- Comandos de voz (Vosk)
- Arquitectura distribuida (cliente-servidor)
- Modo demo con visualización en pantalla

## 🧠 Arquitectura

- Cliente Raspberry Pi / Windows
- Servidor Flask con YOLOv8

## 🛠️ Tecnologías

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Flask
- Vosk
- pyttsx3 / espeak-ng

## ▶️ Ejecución

Servidor:
python server/server.py

Cliente:
python client/raspberry/paso3.py

##  Notas

- Los modelos .pt no se incluyen por su tamaño.
- Configurar IP del servidor correctamente, recordar que esto es un proyecto pequeño, usamos droidcam pero pueden usar su camara nativa y ya.

##  Autor

Sebastián Zambrano
