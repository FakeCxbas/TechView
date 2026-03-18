import requests

url = "http://192.168.100.9:8000/detect"
image_path = "bus.jpg"

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.json())
