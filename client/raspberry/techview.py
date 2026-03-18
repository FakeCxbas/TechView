# techview_windows.py
import cv2
import time
import threading
import requests
import json
import queue
import os
import numpy as np
import sounddevice as sd
import pyttsx3
from vosk import Model, KaldiRecognizer
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

IS_WINDOWS = True

# ==============================
# CONFIG
# ==============================
SERVER_URL = "http://10.169.208.67:8000/detect" # CONSTE QUE ESTA IP ES POR QUE YA ESTAMOS EN FERIA
VOSK_MODEL_PATH = "models/vosk-model-small-es-0.42"

CAM_W, CAM_H = 640, 480
DET_W, DET_H = 320, 240
SCALE_X = CAM_W / DET_W
SCALE_Y = CAM_H / DET_H

CAMERA_INDEX = 0

# ==============================
# FLAGS
# ==============================
modo_demo = False
modo_alerta = "voz"
perfil = "interior"
modo_noche = False

ultimo_evento = None
ultimo_aviso_tiempo = 0.0

COOLDOWN_INTERIOR = 6.0
COOLDOWN_EXTERIOR = 2.5
COOLDOWN_NOCHE_MULT = 1.8

PRIORIDAD = {"vehiculo": 3, "persona": 2, "interior": 1, "otro": 0}

# ==============================
# AUDIO TTS
# ==============================
engine = pyttsx3.init()
engine.setProperty("rate", 165)

audio_q = queue.Queue()

def hablar(texto):
    engine.say(texto)
    engine.runAndWait()

def audio_worker():
    while True:
        texto = audio_q.get()
        try:
            hablar(texto)
        except:
            pass
        audio_q.task_done()

threading.Thread(target=audio_worker, daemon=True).start()

# ==============================
# TRACKING (ajuste)
# ==============================
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    canonical: str
    last_seen: float
    hits: int = 1
    lost: int = 0

class SimpleTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        now = time.time()
        self.tracks.clear()
        for d in detections:
            x1 = int(d["x1"] * SCALE_X)
            y1 = int(d["y1"] * SCALE_Y)
            x2 = int(d["x2"] * SCALE_X)
            y2 = int(d["y2"] * SCALE_Y)
            self.tracks.append(
                Track(self.next_id, (x1,y1,x2,y2), d["label"], now)
            )
            self.next_id += 1
        return self.tracks

tracker = SimpleTracker()

# ==============================
# DETECCIÓN SERVER
# ==============================
ultima_deteccion = []
lock = threading.Lock()

def deteccion_worker():
    global ultima_deteccion
    while True:
        if frame_det is None:
            time.sleep(0.05)
            continue
        _, jpg = cv2.imencode(".jpg", frame_det)
        try:
            r = requests.post(SERVER_URL, files={"image": jpg.tobytes()}, timeout=0.8)
            if r.status_code == 200:
                with lock:
                    ultima_deteccion = r.json()
        except:
            pass
        time.sleep(0.06)

frame_det = None
threading.Thread(target=deteccion_worker, daemon=True).start()

# ==============================
# CÁMARA
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

print("TechView Windows iniciado")

# ==============================
# LOOP PRINCIPAL
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_det = cv2.resize(frame, (DET_W, DET_H))

    with lock:
        dets = ultima_deteccion.copy()

    tracks = tracker.update(dets)

    for t in tracks:
        x1,y1,x2,y2 = t.bbox
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,t.canonical,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    cv2.imshow("TechView", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()