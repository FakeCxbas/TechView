import cv2
import requests
import time

SERVER_URL = "http://192.168.100.9:8000/detect"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: No se pudo abrir la cámara USB")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: No se pudo leer frame")
        break

    # Reducir resolución (MUY importante para rendimiento)
    frame = cv2.resize(frame, (640, 480))

    # Guardar frame temporal
    cv2.imwrite("frame.jpg", frame)

    # Enviar al servidor
    with open("frame.jpg", "rb") as img:
        files = {"image": img}
        response = requests.post(SERVER_URL, files=files)
        detections = response.json()

    print(detections)

    time.sleep(0.5)  # 2 FPS (ideal para asistencia)

cap.release()
