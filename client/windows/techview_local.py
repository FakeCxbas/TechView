# =========================================================================
#  TECHVIEW SIMPLE PARA MOSTRAR DESDE WINDOWS NATIVO EN CASO DE EMERGENCIA
# =========================================================================
import cv2
import time
import threading
import queue
import pyttsx3
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
CAM_W, CAM_H = 640, 480
CAMERA_INDEX = 0
CONF_THRES = 0.4

# ==============================
# FLAGS
# ==============================
modo_demo = True
modo_alerta = "voz"   # voz | silencio
perfil = "interior"   # interior | exterior

ultimo_evento = None
ultimo_tiempo = 0.0
COOLDOWN = 4.0

# ==============================
# TTS AQUI SI FUE POSIBLEEE
# ==============================
engine = pyttsx3.init()
engine.setProperty("rate", 165)

audio_q = queue.Queue()

def decir(texto):
    engine.say(texto)
    engine.runAndWait()

def audio_worker():
    while True:
        texto = audio_q.get()
        try:
            decir(texto)
        except:
            pass
        audio_q.task_done()

threading.Thread(target=audio_worker, daemon=True).start()

def hablar(msg):
    if modo_alerta == "voz":
        audio_q.put(msg)

# ==============================
# MODELO YOLO
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# CÁMARA
# ==============================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

print("TechView LOCAL iniciado (sin server)")

# ==============================
# LOOP PRINCIPAL
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, conf=CONF_THRES, verbose=False)[0]

    evento_actual = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # colores básicos
        color = (0, 255, 0)
        if label in ["car", "bus", "truck", "motorcycle"]:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # evento simple (prioridad)
        if label in ["person", "car", "bus", "truck", "motorcycle"]:
            evento_actual = label

    # ==========================
    # ALERTA SIMPLE
    # ==========================
    ahora = time.time()
    if evento_actual and evento_actual != ultimo_evento:
        if ahora - ultimo_tiempo > COOLDOWN:
            if evento_actual == "person":
                hablar("Persona detectada")
            else:
                hablar("Vehículo detectado")

            ultimo_evento = evento_actual
            ultimo_tiempo = ahora

    cv2.putText(
        frame,
        "TECHVIEW LOCAL | YOLO | SIN SERVER",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("TechView", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()