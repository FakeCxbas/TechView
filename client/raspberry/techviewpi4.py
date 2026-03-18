# ===================================================================================
#  DESARROLLO DE GAFAS INTELIGENTES PARA AYUDAR A PERSONAS DE NULA O REDUCIDA VISION
# =================================================================================== 
import cv2
import time
import threading
import requests
import json
import subprocess
import queue
import os
IS_WINDOWS = os.name == "nt"
from vosk import Model, KaldiRecognizer
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ==============================
# CONFIG
# ==============================
SERVER_URL = "http://10.169.208.67:8000/detect" #IP YA EN LA CALLE
VOSK_MODEL_PATH = "models/vosk-model-small-es-0.42"

CAM_W, CAM_H = 640, 480
DET_W, DET_H = 320, 240

SCALE_X = CAM_W / DET_W
SCALE_Y = CAM_H / DET_H

CAMERA_INDEX = 0 if IS_WINDOWS else "/dev/video0"
AUDIO_DEVICE = "plughw:Loopback,1,0"


# ==============================
# FLAGS / MODOS
# ==============================
modo_demo = False

# modos de alerta: "silencio" | "pitidos" | "voz"
modo_alerta = "silencio"

# perfil: "interior" | "exterior"
perfil = "interior"

# día/noche
modo_noche = False

# anti-spam
ultimo_evento = None
ultimo_aviso_tiempo = 0.0

COOLDOWN_INTERIOR = 6.0
COOLDOWN_EXTERIOR = 2.5
COOLDOWN_NOCHE_MULT = 1.8

# prioridad: vehículos > personas > otros
PRIORIDAD = {"vehiculo": 3, "persona": 2, "interior": 1, "otro": 0}

# ==============================
# DETECCIÓN (compartida con el server)
# ==============================
frame_para_deteccion = None
ultima_deteccion = []
lock_deteccion = threading.Lock()

# ==============================
# AUDIO QUEUE (voz/pitidos en orden)
# ==============================
audio_q: "queue.Queue[tuple]" = queue.Queue(maxsize=100)

def enqueue_say(texto: str):
    try:
        audio_q.put_nowait(("say", texto))
    except queue.Full:
        pass

def enqueue_beep(kind: str):
    try:
        audio_q.put_nowait(("beep", kind))
    except queue.Full:
        pass

def audio_worker():
    while True:
        kind, payload = audio_q.get()
        try:
            if kind == "say":
                hablar_bloqueante(payload)
            elif kind == "beep":
                beep_pattern(payload)
        except:
            pass
        audio_q.task_done()

# ==============================
# TTS (ESPEAK afinado, bloqueante)
# ==============================
def hablar_bloqueante(texto: str):
    s = "120"
    a = "170"
    p = "38"
    k = "7"
    g = "8"

    if modo_noche:
        s = "115"
        a = "130"
        p = "36"
        k = "6"
        g = "10"

    subprocess.run(
        ["espeak-ng", "-v", "es", "-s", s, "-p", p, "-a", a, "-k", k, "-g", g, texto],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ==============================
# PITIDOS (sintéticos)
# ==============================
def beep(freq=1200, dur=0.08):
    subprocess.run(
        [
            "speaker-test",
            "-t", "sine",
            "-f", str(freq),
            "-l", "1",
            "-P", "1"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=dur
    )

def beep_pattern(kind: str):
    if kind == "vehiculo":
        beep(1500, 0.07)
        time.sleep(0.05)
        beep(1500, 0.07)
    elif kind == "persona":
        beep(1000, 0.09)

# ===========================================
# LABELS: normalización + display en español
# ===========================================
# Canonical keys (internos) -> display (español)
# FIXEADO POR FIN, A FINAL NO PUDIMOS PONER TTS
DISPLAY_ES = {
    "person": "persona",
    "car": "auto",
    "bus": "bus",
    "truck": "camión",
    "motorbike": "moto",
    "bicycle": "bicicleta",

    # interior comunes (COCO)
    "chair": "silla",
    "couch": "sofá",
    "sofa": "sofá",
    "bed": "cama",
    "dining table": "mesa",
    "table": "mesa",
    "tv": "televisor",
    "laptop": "laptop",
    "keyboard": "teclado",
    "mouse": "mouse",
    "microwave": "microondas",
    "refrigerator": "refrigeradora",
    "sink": "lavamanos",
    "toilet": "inodoro",
    "book": "libro",
    "bottle": "botella",
    "cup": "taza",

    # exteriores “típicos” (si tu dataset los tiene)
    "post": "poste",
    "pole": "poste",
    "traffic light": "semáforo",
}

# Alias desde cualquier cosa a canonical key
ALIAS = {
    # persona
    "persona": "person",
    "person": "person",

    # vehículos
    "auto": "car",
    "carro": "car",
    "car": "car",
    "bus": "bus",
    "camion": "truck",
    "camión": "truck",
    "truck": "truck",
    "moto": "motorbike",
    "motorbike": "motorbike",
    "bicicleta": "bicycle",
    "bicycle": "bicycle",

    # interior
    "silla": "chair",
    "chair": "chair",
    "sofa": "sofa",
    "sofá": "sofa",
    "couch": "couch",
    "cama": "bed",
    "bed": "bed",
    "mesa": "table",
    "table": "table",
    "televisor": "tv",
    "tv": "tv",
    "inodoro": "toilet",
    "toilet": "toilet",
    "microondas": "microwave",
    "microwave": "microwave",
    "refrigeradora": "refrigerator",
    "refrigerator": "refrigerator",
}

def normalize_label(raw: str) -> str:
    raw = (raw or "").strip().lower()
    # algunas detecciones vienen con espacios raros o mayúsculas, intento de fix
    raw = " ".join(raw.split())
    return ALIAS.get(raw, raw)

def display_label_es(canonical: str) -> str:
    return DISPLAY_ES.get(canonical, canonical)

def tipo_objeto(canonical: str) -> str:
    if canonical in ["car", "bus", "truck", "motorbike"]:
        return "vehiculo"
    if canonical == "person":
        return "persona"
    # interior típico (COCO)
    if canonical in ["chair", "sofa", "couch", "bed", "table", "tv", "toilet", "sink", "refrigerator", "microwave"]:
        return "interior"
    return "otro"

# ==============================
# COLORES POR CLASE / GRUPO
# ==============================
# BGR (OpenCV)
COLOR_BY_TYPE = {
    "vehiculo": (0, 0, 255),    # rojo
    "persona": (0, 255, 0),     # verde
    "interior": (255, 0, 0),    # azul
    "otro": (0, 255, 255),      # amarillo
}

# Override por clase (se ve mejorcito)
COLOR_BY_CLASS = {
    "person": (0, 255, 0),
    "car": (0, 0, 255),
    "bus": (0, 0, 255),
    "truck": (0, 0, 255),
    "motorbike": (0, 0, 255),
    "chair": (255, 0, 0),
    "sofa": (255, 0, 0),
    "couch": (255, 0, 0),
    "bed": (255, 0, 0),
    "table": (255, 0, 0),
    "tv": (255, 0, 0),
}

def color_for(canonical: str) -> Tuple[int, int, int]:
    if canonical in COLOR_BY_CLASS:
        return COLOR_BY_CLASS[canonical]
    return COLOR_BY_TYPE.get(tipo_objeto(canonical), (200, 200, 200))

# ==============================
# UTILIDADES: lado/cercanía
# ==============================
def lado_objeto(x1, x2, frame_w=CAM_W):
    cx = (x1 + x2) / 2.0
    if cx < frame_w * 0.33:
        return "izquierda"
    if cx > frame_w * 0.66:
        return "derecha"
    return "al frente"

def cercania_por_bbox(x1, y1, x2, y2):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    area = w * h
    if area > 90000:
        return "cerca"
    if area > 35000:
        return "medio"
    return "lejos"

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2-ax1)) * max(1, (ay2-ay1))
    area_b = max(1, (bx2-bx1)) * max(1, (by2-by1))
    return inter / float(area_a + area_b - inter + 1e-9)

# ==============================
# TRACKER PRO (IoU + smoothing)
# ==============================
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    canonical: str
    last_seen: float
    hits: int = 1
    lost: int = 0
    label_votes: Dict[str, int] = field(default_factory=dict)

    def update_label(self, canonical: str):
        self.label_votes[canonical] = self.label_votes.get(canonical, 0) + 1
        # label estable por mayoría
        self.canonical = max(self.label_votes.items(), key=lambda kv: kv[1])[0]

class SimpleTracker:
    def __init__(self, iou_thresh=0.25, max_lost=12, alpha=0.7):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.alpha = alpha
        self.next_id = 1
        self.tracks: List[Track] = []

    def update(self, detections: List[Dict]) -> List[Track]:
        now = time.time()

        det_bboxes = []
        det_labels = []
        for d in detections:
            # d viene en coords de DET (320x240) -> ya escalamos aquí
            x1 = int(d.get("x1", 0) * SCALE_X)
            y1 = int(d.get("y1", 0) * SCALE_Y)
            x2 = int(d.get("x2", 0) * SCALE_X)
            y2 = int(d.get("y2", 0) * SCALE_Y)
            det_bboxes.append((x1, y1, x2, y2))
            det_labels.append(normalize_label(d.get("label", "")))

        # marcar tracks como perdidos inicialmente
        for t in self.tracks:
            t.lost += 1

        # matching greedy por IoU
        used_det = set()
        for t in self.tracks:
            best_iou = 0.0
            best_j = -1
            for j, bb in enumerate(det_bboxes):
                if j in used_det:
                    continue
                score = iou(t.bbox, bb)
                if score > best_iou:
                    best_iou = score
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_thresh:
                used_det.add(best_j)
                # smoothing bbox (reduce salto)
                nx1, ny1, nx2, ny2 = det_bboxes[best_j]
                ox1, oy1, ox2, oy2 = t.bbox
                sx1 = int(self.alpha * ox1 + (1 - self.alpha) * nx1)
                sy1 = int(self.alpha * oy1 + (1 - self.alpha) * ny1)
                sx2 = int(self.alpha * ox2 + (1 - self.alpha) * nx2)
                sy2 = int(self.alpha * oy2 + (1 - self.alpha) * ny2)
                t.bbox = (sx1, sy1, sx2, sy2)

                t.last_seen = now
                t.hits += 1
                t.lost = 0
                t.update_label(det_labels[best_j])

        # crear tracks nuevos para detecciones no usadas
        for j, bb in enumerate(det_bboxes):
            if j in used_det:
                continue
            tr = Track(
                track_id=self.next_id,
                bbox=bb,
                canonical=det_labels[j],
                last_seen=now,
            )
            tr.update_label(det_labels[j])
            self.next_id += 1
            self.tracks.append(tr)

        # limpiar tracks demasiado perdidos
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        return self.tracks

tracker = SimpleTracker(iou_thresh=0.25, max_lost=12, alpha=0.75)

# ======================================================================================================
# UBICACIÓN (no pudimos literalmente doxear a la persona, asi que tenemos que aproximarnos por drodicam)
# ======================================================================================================
def frases_ubicacion_honesta(tracks: List[Track]):
    frases = ["Ubicación aproximada según la red."]

    labels = [t.canonical for t in tracks if t.lost == 0]
    vehiculos = any(c in ["car", "bus", "truck", "motorbike"] for c in labels)
    personas = any(c == "person" for c in labels)

    if vehiculos:
        frases += ["Zona urbana.", "Tránsito cercano."]
    elif personas:
        frases += ["Zona urbana."]
    else:
        frases += ["Zona tranquila."]

    return frases

# ==========================================
#  DETECCIÓN PRO (constante + configurable)
# ==========================================
def cooldown_actual():
    base = COOLDOWN_EXTERIOR if perfil == "exterior" else COOLDOWN_INTERIOR
    if modo_noche:
        base *= COOLDOWN_NOCHE_MULT
    return base

def elegir_objetivo(tracks: List[Track]):
    best = None
    best_score = -1

    for t in tracks:
        if t.lost != 0:
            continue
        canon = t.canonical
        tipo = tipo_objeto(canon)

        x1, y1, x2, y2 = t.bbox
        cerc = cercania_por_bbox(x1, y1, x2, y2)
        lado = lado_objeto(x1, x2)

        tipo_score = PRIORIDAD.get(tipo, 0)
        cerc_score = {"lejos": 0, "medio": 1, "cerca": 2}[cerc]

        # interior: vehículos menos agresivos (por si TV/imagen), salvo “cerca”
        if perfil == "interior" and tipo == "vehiculo" and cerc != "cerca":
            tipo_score = 1

        score = tipo_score * 10 + cerc_score
        if score > best_score:
            best_score = score
            best = (tipo, cerc, lado, canon)

    return best  # (tipo, cerc, lado, canonical)


    # En modo interior, siempre voz (no pitidos)
    if perfil == "interior":
       modo_alerta_actual = "voz"
    else:
       modo_alerta_actual = modo_alerta

    if modo_alerta_actual == "pitidos":
       enqueue_beep("vehiculo" if tipo == "vehiculo" else "persona")
       return


    tipo, cerc, lado, canon = objetivo

    # interior: no spamear interior/otros automáticamente
    if perfil == "interior" and tipo in ["interior", "otro"]:
        return

    # interior: vehículos solo si cerca
    if perfil == "interior" and tipo == "vehiculo" and cerc != "cerca":
        return

    ahora = time.time()
    if ahora - ultimo_aviso_tiempo < cooldown_actual():
        return

    evento = f"{tipo}:{cerc}:{lado}:{canon}"
    if evento == ultimo_evento:
        return

    ultimo_evento = evento
    ultimo_aviso_tiempo = ahora

    if modo_alerta == "pitidos":
        if tipo == "vehiculo":
            enqueue_beep("vehiculo")
        else:
            enqueue_beep("persona")
        return

    # modo voz: frases cortas, claras
    if tipo == "vehiculo":
        if cerc == "cerca":
            enqueue_say(f"Vehículo cerca, {lado}.")
        elif cerc == "medio":
            enqueue_say(f"Vehículo, {lado}.")
        else:
            enqueue_say("Vehículo detectado.")
    elif tipo == "persona":
        if cerc == "cerca":
            enqueue_say(f"Persona cerca, {lado}.")
        else:
            enqueue_say(f"Persona, {lado}.")

#================================
#   NUEVO CICLO DE DETECCION
#================================
def deteccion_pro_tick(tracks):
    global ultimo_evento, ultimo_aviso_tiempo

    if modo_alerta == "silencio":
        return

    ahora = time.time()
    if ahora - ultimo_aviso_tiempo < cooldown_actual():
        return

    # Elegir el objeto más relevante
    objetivo = elegir_objetivo(tracks)
    if objetivo is None:
        return

    tipo, cerc, lado, canon = objetivo
    nombre = display_label_es(canon)

    # Identificador de evento (anti-spam)
    evento = f"{canon}:{cerc}:{lado}"
    if evento == ultimo_evento:
        return

    ultimo_evento = evento
    ultimo_aviso_tiempo = ahora

    # =========================
    # PERFIL INTERIOR
    # =========================
    if perfil == "interior":
        # En interior SIEMPRE voz, y de TODO lo cercano
        if cerc == "cerca":
            enqueue_say(f"{nombre} cerca, {lado}.")
        else:
            enqueue_say(f"{nombre}, {lado}.")
        return

    # =========================
    # PERFIL EXTERIOR
    # =========================
    if modo_alerta == "pitidos":
        if tipo == "vehiculo":
            enqueue_beep("vehiculo")
        else:
            enqueue_beep("persona")
        return

    # Voz exterior (solo lo importante)
    if tipo == "vehiculo":
        if cerc == "cerca":
            enqueue_say(f"Vehículo cerca, {lado}.")
        elif cerc == "medio":
            enqueue_say(f"Vehículo, {lado}.")
    elif tipo == "persona":
        if cerc == "cerca":
            enqueue_say(f"Persona cerca, {lado}.")



# ========================================================
# DETECCIÓN CUANDO SE LO SOLITICA | COMANDO: ("detectar")
# ========================================================
def frases_deteccion_bajo_demanda(tracks: List[Track]):
    # cuenta por categorías y además menciona el objetivo más importante
    actives = [t for t in tracks if t.lost == 0]

    labels = [t.canonical for t in actives]
    n_personas = sum(1 for c in labels if c == "person")
    n_veh = sum(1 for c in labels if c in ["car", "bus", "truck", "motorbike"])

    # interior/otros: lo reportamos si estás en casa (sirve mucho)
    n_interior = sum(1 for c in labels if tipo_objeto(c) == "interior")

    if n_personas == 0 and n_veh == 0 and n_interior == 0:
        return ["Zona tranquila.", "Sin obstáculos cercanos."]

    frases = []
    if n_personas == 1:
        frases.append("Una persona detectada.")
    elif n_personas > 1:
        frases.append("Varias personas detectadas.")

    if n_veh > 0:
        frases.append("Tránsito detectado.")

    if perfil == "interior" and n_interior > 0:
        frases.append("Objetos de interior detectados.")

    obj = elegir_objetivo(tracks)
    if obj:
        tipo, cerc, lado, canon = obj
        nombre = display_label_es(canon)
        if tipo == "vehiculo":
            frases.append(f"Prioridad: {nombre}, {cerc}, {lado}.")
        elif tipo == "persona":
            frases.append(f"Prioridad: persona, {cerc}, {lado}.")
        elif tipo == "interior":
            frases.append(f"Al frente: {nombre}.")
    return frases[:3]

# ==============================
# VOZ – ALSA + VOSK
# ==============================
def voice_listener():
    global modo_demo, modo_alerta, perfil, modo_noche

    try:
        model = Model(VOSK_MODEL_PATH)
        rec = KaldiRecognizer(model, 16000)
    except Exception as e:
        print("ERROR Vosk:", e)
        return

    cmd = [
        "arecord",
        "-D", AUDIO_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-q"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        print("🎙️ Escuchando comandos de voz")
    except Exception as e:
        print("ERROR arecord:", e)
        return

    ultimo_tiempo = 0.0
    COOLDOWN_CMD = 1.2

    while True:
        data = proc.stdout.read(8000)
        if not data:
            continue

        if rec.AcceptWaveform(data):
            texto = json.loads(rec.Result()).get("text", "").lower().strip()
            if not texto:
                continue

            ahora = time.time()
            if ahora - ultimo_tiempo < COOLDOWN_CMD:
                continue

            print("VOZ:", texto)

            # ========= ALERTA =========
            if "modo voz" in texto:
                modo_alerta = "voz"
                enqueue_say("Modo voz activado.")
                ultimo_tiempo = ahora

            elif "modo pitidos" in texto or "modo pitido" in texto:
                modo_alerta = "pitidos"
                enqueue_say("Modo pitidos activado.")
                ultimo_tiempo = ahora

            elif "modo silencio" in texto:
                modo_alerta = "silencio"
                enqueue_say("Modo silencio activado.")
                ultimo_tiempo = ahora

            # ========= PERFIL =========
            elif "modo exterior" in texto or "perfil exterior" in texto:
                perfil = "exterior"
                enqueue_say("Perfil exterior activado.")
                ultimo_tiempo = ahora

            elif "modo interior" in texto or "perfil interior" in texto:
                perfil = "interior"
                enqueue_say("Perfil interior activado.")
                ultimo_tiempo = ahora

            # ========= DÍA/NOCHE =========
            elif "modo noche" in texto:
                modo_noche = True
                enqueue_say("Modo noche activado.")
                ultimo_tiempo = ahora

            elif "modo dia" in texto or "modo día" in texto:
                modo_noche = False
                enqueue_say("Modo día activado.")
                ultimo_tiempo = ahora

            # ========= PRINCIPALES =========
            elif "deteccion" in texto or "detectar" in texto:
                with lock_deteccion:
                    dets = ultima_deteccion.copy()
                tracks = tracker.update(dets)
                for f in frases_deteccion_bajo_demanda(tracks):
                    enqueue_say(f)
                ultimo_tiempo = ahora

            elif any(p in texto for p in ["donde", "dónde", "ubicacion", "ubicación", "lugar"]):
                with lock_deteccion:
                    dets = ultima_deteccion.copy()
                tracks = tracker.update(dets)
                for f in frases_ubicacion_honesta(tracks):
                    enqueue_say(f)
                ultimo_tiempo = ahora

            elif "hora" in texto:
                hora = datetime.now().strftime("%H:%M")
                enqueue_say(f"Son las {hora}.")
                ultimo_tiempo = ahora

            elif "estado" in texto:
                enqueue_say("Sistema activo.")
                enqueue_say(f"Perfil {perfil}.")
                enqueue_say(f"Alerta {modo_alerta}.")
                if modo_noche:
                    enqueue_say("Modo noche.")
                ultimo_tiempo = ahora

            # ========= DEMO =========
            elif "modo demo" in texto or "modo demostracion" in texto or "modo demostración" in texto:
                modo_demo = True
                enqueue_say("Modo demostración activado.")
                ultimo_tiempo = ahora

            elif "salir demo" in texto or "quitar demo" in texto:
                modo_demo = False
                enqueue_say("Modo demostración desactivado.")
                ultimo_tiempo = ahora

# ==============================
# DETECCIÓN – HILO (server)
# ==============================
def deteccion_worker():
    global ultima_deteccion

    last_print = 0.0
    seen_labels = set()

    while True:
        try:
            if frame_para_deteccion is None:
                time.sleep(0.03)
                continue

            _, img_encoded = cv2.imencode(".jpg", frame_para_deteccion, [cv2.IMWRITE_JPEG_QUALITY, 50])

            response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()}, timeout=0.8)

            if response.status_code == 200:
                data = response.json()

                # guardar labels crudos para diagnóstico
                for d in data:
                    raw = (d.get("label", "") or "").strip()
                    if raw:
                        seen_labels.add(raw)

                    # normaliza para el resto del sistema
                    d["label"] = normalize_label(d.get("label", ""))

                with lock_deteccion:
                    ultima_deteccion = data

            # imprime labels vistos cada 15s
            now = time.time()
            if now - last_print > 15:
                if seen_labels:
                    print("LABELS vistos:", sorted(list(seen_labels))[:50])
                last_print = now
                seen_labels.clear()

        except:
            pass

        time.sleep(0.06)

# ==============================
# START THREADS
# ==============================
threading.Thread(target=audio_worker, daemon=True).start()
threading.Thread(target=voice_listener, daemon=True).start()
threading.Thread(target=deteccion_worker, daemon=True).start()

# ==============================
# CAMERA
# ==============================
if IS_WINDOWS:
    cap = cv2.VideoCapture(CAMERA_INDEX)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

print("TechView iniciado")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_para_deteccion = cv2.resize(frame, (DET_W, DET_H))

    with lock_deteccion:
        detecciones = ultima_deteccion.copy()

    # tracking (suaviza saltos)
    tracks = tracker.update(detecciones)

    # detección pro constante (no bloquea)
    deteccion_pro_tick(tracks)

    # demo: dibuja tracks (no detección cruda)
    if modo_demo:
        for t in tracks:
            if t.lost != 0:
                continue
            x1, y1, x2, y2 = t.bbox
            canon = t.canonical
            typ = tipo_objeto(canon)
            cerc = cercania_por_bbox(x1, y1, x2, y2)
            lado = lado_objeto(x1, x2)

            col = color_for(canon)
            name = display_label_es(canon)

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                frame,
                f"ID {t.track_id} | {name} | {cerc} | {lado}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                col,
                2
            )

        cv2.putText(
            frame,
            f"DEMO | perfil:{perfil} | alerta:{modo_alerta} | noche:{'si' if modo_noche else 'no'}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("TechView", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()