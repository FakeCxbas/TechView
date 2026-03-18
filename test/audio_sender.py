import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import tempfile
import os

SERVER_URL = "http://192.168.100.9:8001/voice"

DEVICE_ID = 3          # USB Audio
DURATION = 3           # segundos
SAMPLE_RATE = 44100    # nativo del mic

print("🎤 Audio sender listo")

while True:
    try:
        input("▶ Presiona ENTER y habla (3 segundos)...")

        print("🎙️ Grabando...")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="int16",
            device=DEVICE_ID
        )
        sd.wait()

        # 👉 Convertir a MONO
        audio_mono = audio.mean(axis=1).astype(np.int16)

        # 👉 Guardar WAV limpio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, SAMPLE_RATE, audio_mono)
            audio_path = f.name

        # 👉 Enviar al server
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            r = requests.post(SERVER_URL, files=files)

        os.remove(audio_path)

        print("📡 Server respondió:", r.json())

    except KeyboardInterrupt:
        print("\n👋 Saliendo.")
        break
