import requests

SERVER_URL = "http://192.168.100.9:8000/detect"
IMAGE_PATH = "bus.jpg"

with open(IMAGE_PATH, "rb") as img:
    response = requests.post(SERVER_URL, files={"image": img})
    detections = response.json()

for d in detections:
    print(d)
