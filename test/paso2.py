import cv2
import time
import threading
import queue
import requests
import numpy as np
import sounddevice as sd
import vosk
import json
import pyttsx3

# =========================
# CONFIG
# =========================
SERVER_URL = "http://192.168.100.9:8000/detect"
CAMERA_INDEX = 0

# =========================
# AUDIO / VOZ
# =========================
engine = pyttsx3.init()
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text:
            engine.say(text)
            engine.runAndWait()

def enqueue_say(text):
    tts_queue.put(text)

threading.Thread(target=tts_worker, daemon=True).start()

# =========================
# VOZ (VOSK)
# =========================
model = vosk.Model("model")
rec = vosk.KaldiRecognizer(model, 16000)

audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_q.put(bytes(indata))

def voice_worker():
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            data = audio_q.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                texto = res.get("text", "")
                if texto:
                    procesar_comando(texto)

threading.Thread(target=voice_worker, daemon=True).start()

# =========================
# MODOS
# =========================
modo_demo = False
modo_aistente = False  # 👈 TU VARIABLE ORIGINAL

# =========================
# ESTADO ASISTENTE (NUEVO, BIEN INTEGRADO)
# =========================
asistente_estado = {
    "tipo": None,
    "lado": None,
    "ultimo": 0.0
}

# =========================
# UTILS
# =========================
def lado_objeto(x1, x2, w=640):
    cx = (x1 + x2) / 2
    if cx < w * 0.4:
        return "izquierda"
    elif cx > w * 0.6:
        return "derecha"
    return "frente"

def tipo_objeto(label):
    if label in ["car", "truck", "bus", "motorcycle"]:
        return "vehiculo"
    if label == "person":
        return "persona"
    return None

def cercania_por_bbox(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)
    return "cerca" if area > 40000 else "lejos"

# =========================
# VOZ → COMANDOS
# =========================
def procesar_comando(texto):
    global modo_demo, modo_aistente

    print("VOZ:", texto)

    if "modo asistente" in texto:
        modo_aistente = True
        enqueue_say("Modo asistente activado")

    elif "salir asistente" in texto:
        modo_aistente = False
        enqueue_say("Modo asistente desactivado")

    elif "modo demo" in texto:
        modo_demo = True
        modo_aistente = False
        enqueue_say("Modo demo")

# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

print("TechView iniciado")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape

    # =========================
    # SERVER DETECTION
    # =========================
    _, jpg = cv2.imencode(".jpg", frame)
    try:
        r = requests.post(SERVER_URL, files={"image": jpg.tobytes()}, timeout=0.5)
        detecciones = r.json()
    except:
        detecciones = []

    # =========================
    # DIBUJO DEMO
    # =========================
    if modo_demo:
        for d in detecciones:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # =========================
    # MODO ASISTENTE ORIENTADO (CORREGIDO)
    # =========================
    if modo_aistente:
        ahora = time.time()
        peligro = None

        for d in detecciones:
            tipo = tipo_objeto(d["label"])
            if tipo not in ["vehiculo", "persona"]:
                continue

            x1, y1, x2, y2 = d["bbox"]
            if cercania_por_bbox(x1, y1, x2, y2) != "cerca":
                continue

            lado = lado_objeto(x1, x2, w)
            peligro = (tipo, lado)
            break

        if peligro:
            tipo, lado = peligro
            cambio = (
                asistente_estado["tipo"] != tipo or
                asistente_estado["lado"] != lado
            )

            if cambio and ahora - asistente_estado["ultimo"] > 1.5:
                enqueue_say(f"{tipo} a la {lado}" if lado != "frente" else f"{tipo} al frente")
                asistente_estado.update({
                    "tipo": tipo,
                    "lado": lado,
                    "ultimo": ahora
                })
        else:
            if asistente_estado["tipo"] is not None:
                enqueue_say("Camino libre")
                asistente_estado.update({
                    "tipo": None,
                    "lado": None,
                    "ultimo": ahora
                })

    cv2.imshow("TechView", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
