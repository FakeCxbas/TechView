import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print(f"Nivel audio: {volume_norm:.2f}")

print("Hablá al micrófono...")
with sd.InputStream(device=3, channels=1, samplerate=16000, callback=callback):
    input("Presiona Enter para salir\n")
