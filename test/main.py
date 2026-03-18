import cv2
import time
import requests
import subprocess
import signal
import sys

# =========================
# CONFIGURACIÓN
# =========================

SERVER_URL = "http://192.168.100.9:8000/detect"
CAM_INDEX = 0  # DroidCam

# Resolución de captura (AJUSTE 1)
CAP_W, CAP_H = 640, 360

# Resolución de display (AJUSTE 2)
DISP_W, DISP_H = 640, 360

# FPS del display (AJUSTE 3)
DISPLAY_FPS = 15
ultimo_display = 0.0

# Detección
FRAME_SKIP = 5
frame_count = 0
detecciones_cache = []

# Umbrales
AREA_MEDIO = 0.12
AREA_CERCA = 0.20

# Beep
BEEP_LENTO = 1.2
BEEP_RAPIDO = 0.4

# =========================
# CIERRE LIMPIO
# =========================

def cerrar(sig, frame):
    print("\nCerrando sistema...")
    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cerrar)
signal.signal(signal.SIGTERM, cerrar)

# =========================
# TTS LOCAL (PIPER)
# =========================

PIPER_BIN = "/home/pi/.local/bin/piper"
PIPER_MODEL = "/home/pi/piper/es_ES-carlfm-x_low.onnx"
PIPER_CONFIG = "/home/pi/piper/es_ES-carlfm-x_low.onnx.json"

def hablar(texto: str):
    try:
        p = subprocess.Popen(
            [
                PIPER_BIN,
                "--model", PIPER_MODEL,
                "--config", PIPER_CONFIG,
                "--output-file", "/tmp/voz.wav"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        p.communicate(texto)
        subprocess.run(["aplay", "/tmp/voz.wav"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except:
        pass

# =========================
# BEEP
# =========================

def beep():
    subprocess.run(
        ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# =========================
# UTILIDADES
# =========================

def lado_objeto(x1, x2, ancho):
    c = (x1 + x2) / 2
    if c < ancho * 0.4:
        return "izquierda"
    elif c > ancho * 0.6:
        return "derecha"
    else:
        return "frente"

def cercania_por_bbox(x1, y1, x2, y2, w, h):
    area = ((x2 - x1) * (y2 - y1)) / (w * h)
    if area > AREA_CERCA:
        return "cerca"
    elif area > AREA_MEDIO:
        return "medio"
    else:
        return "lejos"

# =========================
# ESTADO
# =========================

estado_actual = "despejado"
estado_anterior = None
ultimo_beep = 0.0

# =========================
# CÁMARA
# =========================

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    sys.exit(1)

# AJUSTE 1: bajar resolución de captura
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

cv2.namedWindow("TechView - Orientación", cv2.WINDOW_NORMAL)

print("🟢 Sistema iniciado")
hablar("Sistema iniciado.")

# =========================
# LOOP PRINCIPAL
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    frame_count += 1

    # 🔁 Detección cada N frames (no bloquea video)
    if frame_count % FRAME_SKIP == 0:
        try:
            _, img = cv2.imencode(".jpg", frame)
            r = requests.post(
                SERVER_URL,
                files={"image": img.tobytes()},
                timeout=1.0
            )
            detecciones_cache = r.json()
        except:
            detecciones_cache = []

    detecciones = detecciones_cache

    obstaculo_frontal = False
    cercania_max = "lejos"
    direccion = None

    # Dibujar y analizar
    for d in detecciones:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        label = d["label"]

        lado = lado_objeto(x1, x2, w)
        cercania = cercania_por_bbox(x1, y1, x2, y2, w, h)

        color = (0, 255, 0)
        if cercania == "medio":
            color = (0, 255, 255)
        elif cercania == "cerca":
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {cercania}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if lado == "frente" and cercania in ("medio", "cerca"):
            obstaculo_frontal = True
            cercania_max = cercania
            direccion = "izquierda"

    estado_actual = "obstaculo" if obstaculo_frontal else "despejado"

    # Voz por cambio de estado
    if estado_actual != estado_anterior:
        if estado_actual == "despejado":
            hablar("Camino despejado.")
        else:
            beep()
            if cercania_max == "cerca":
                hablar("Obstáculo detectado. Deténgase.")
            else:
                hablar(f"Obstáculo detectado. Muévase a la {direccion}.")
        estado_anterior = estado_actual

    # Beep progresivo
    ahora = time.time()
    if estado_actual == "obstaculo":
        intervalo = BEEP_RAPIDO if cercania_max == "cerca" else BEEP_LENTO
        if ahora - ultimo_beep > intervalo:
            beep()
            ultimo_beep = ahora

    # =========================
    # DISPLAY OPTIMIZADO (AJUSTES 2 y 3)
    # =========================
    ahora = time.time()
    if ahora - ultimo_display >= 1.0 / DISPLAY_FPS:
        frame_display = cv2.resize(frame, (DISP_W, DISP_H))
        cv2.imshow("TechView - Orientación", frame_display)
        ultimo_display = ahora

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        cerrar(None, None)
