from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)

print("🧠 Cargando Whisper...")
model = whisper.load_model("small")
print("✅ Whisper listo")

def interpretar_comando(texto: str):
    texto = texto.lower()

    if "techview" not in texto:
        return None

    if "modo demostración" in texto or "modo demostracion" in texto:
        return "modo_demostracion"

    if "desactivar demostración" in texto or "desactivar demostracion" in texto:
        return "salir_demostracion"

    if "silenciar" in texto:
        return "voz_off"

    if "activar voz" in texto:
        return "voz_on"

    return None


@app.route("/voice", methods=["POST"])
def voice():
    if "audio" not in request.files:
        return jsonify({"error": "no audio"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    print("🎧 Audio recibido, transcribiendo...")
    result = model.transcribe(audio_path, language="es")
    os.remove(audio_path)

    text = result["text"].strip()
    print("🧠 Whisper oyó:", text)

    comando = interpretar_comando(text)

    return jsonify({
        "text": text,
        "command": comando
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)
