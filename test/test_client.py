import requests

url = "http://127.0.0.1:8000/detect"
image_path = "bus.jpg"

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.json())
