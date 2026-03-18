# PRUEBA 
import cv2
import time
import threading
import subprocess
import json
import requests
import math

# ==============================
# CONFIGURACIÓN
# ==============================
SERVER_URL = "http://192.168.1.100:5000/detections"  # CAMBIAR IP
CAM_W = 640
CAM_H = 480
FPS_LIMIT = 30

# FLAGS GLOBALES
modo_demo = False
voz_habilitada = True
anuncios_habilitados = True
ultimo_anuncio = {}
cooldown_anuncio = 3  # segundos

# ==============================
# INICIALIZACIÓN OPENCV (CLAVE)
# ==============================
cv2.startWindowThread()

# ==============================
# FUNCIONES
# ==============================
def decir(texto):
    if not voz_habilitada:
        return
    subprocess.Popen(
        ["pico2wave", "-w", "/tmp/tts.wav", texto],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    subprocess.Popen(
        ["aplay", "/tmp/tts.wav"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def puede_anunciar(label):
    ahora = time.time()
    if label not in ultimo_anuncio:
        ultimo_anuncio[label] = ahora
        return True
    if ahora - ultimo_anuncio[label] >= cooldown_anuncio:
        ultimo_anuncio[label] = ahora
        return True
    return False

def obtener_detecciones():
    try:
        r = requests.get(SERVER_URL, timeout=0.5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []

# ==============================
# CÁMARA
# ==============================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit(1)

print("📷 Cámara abierta correctamente")

# ==============================
# VENTANA (OBLIGATORIA)
# ==============================
cv2.namedWindow("TechView", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TechView", CAM_W, CAM_H)

print("🧠 TechView iniciado")

# ==============================
# LOOP PRINCIPAL
# ==============================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame perdido")
        continue

    ahora = time.time()
    if ahora - prev_time < 1 / FPS_LIMIT:
        continue
    prev_time = ahora

    detecciones = obtener_detecciones()

    for det in detecciones:
        label = det.get("label", "")
        conf = det.get("confidence", 0)
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        texto = f"{label} {int(conf * 100)}%"
        cv2.putText(
            frame,
            texto,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        if anuncios_habilitados and puede_anunciar(label):
            decir(label)

    cv2.imshow("TechView", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ==============================
# SALIDA LIMPIA
# ==============================
cap.release()
cv2.destroyAllWindows()
print("👋 TechView cerrado")
