# server.py
# TechView Server - YOLO + TTS remoto (Windows compatible)

from flask import Flask, request, jsonify, send_file
import os
import threading
import torch
import uuid
import tempfile
import pyttsx3
import numpy as np
import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
CONF_THRESHOLD = 0.6
IOU_NMS = 0.6

TMP_DIR = tempfile.gettempdir()

tts_lock = threading.Lock()

app = Flask(__name__)

# =========================
# TTS (Windows nativo)
# =========================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 175)
tts_engine.setProperty("volume", 1.0)

def generar_tts(texto: str) -> str:
    uid = str(uuid.uuid4())
    wav_path = os.path.join(TMP_DIR, f"{uid}.wav")

    tts_engine.save_to_file(texto, wav_path)
    tts_engine.runAndWait()

    return wav_path

@app.route("/tts", methods=["POST"])
def tts():
    # SOLO UNA VOZ A LA VEZ (evita bloqueo en Windows)
    if not tts_lock.acquire(blocking=False):
        return ("", 204)

    try:
        data = request.get_json(force=True)
        texto = data.get("text", "").strip()

        if not texto:
            return ("", 204)

        wav = generar_tts(texto)
        return send_file(wav, mimetype="audio/wav")

    finally:
        tts_lock.release()


# =========================
# YOLO
# =========================
model = YOLO("yolov8s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify([])

    img = cv2.imdecode(
        np.frombuffer(request.files["image"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return jsonify([])

    results = model.track(
        source=img,
        conf=CONF_THRESHOLD,
        iou=IOU_NMS,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    out = []
    r = results[0]

    if r.boxes is None:
        return jsonify([])

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        out.append({
            "label": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    return jsonify(out)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("🟢 TechView Server iniciado (YOLO + TTS Windows)")
    app.run(host="0.0.0.0", port=8000, debug=False)
