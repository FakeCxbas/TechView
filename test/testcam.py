import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # Fuerza backend correcto para Qt

import cv2

DEVICE = "/dev/video2v4l2-ctl --list-devices
"   # cambia a /dev/video1 si es necesario

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

# Configuración estable para Pi 4
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camara abierta:", cap.isOpened())

cv2.namedWindow("CAMTEST", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame no recibido")
        continue

    cv2.imshow("CAMTEST", frame)

    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
