import cv2
import imutils
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

model_name = "yolov8n_personal.pt"
model = YOLO(model_name)

vehicle_id = 2  # Araç sınıfı (örneğin araba)
track_history = defaultdict(lambda: [])
previous_positions = defaultdict(lambda: None)  # Önceki pozisyonları kaydederiz
speed_history = defaultdict(lambda: [])  # Hız geçmişini kaydederiz

fps = cap.get(cv2.CAP_PROP_FPS)  # Videonun FPS değerini alır
frame_time = 1 / fps  # Her bir kare arasındaki süreyi hesaplar

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=1280)

    # YOLO ile takip
    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")

    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1 + x2) / 2)  # Merkez X
        cy = int((y1 + y2) / 2)  # Merkez Y

        if class_id == vehicle_id:
            # Aracın etrafına dikdörtgen çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Takip edilen nesnenin hızını ve yönünü hesapla
            if previous_positions[track_id] is not None:
                prev_cx, prev_cy = previous_positions[track_id]
                distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                speed = distance / frame_time  # Hız, birim zamanda alınan mesafe

                # Hızı kaydet
                speed_history[track_id].append(speed)
                if len(speed_history[track_id]) > 5:
                    speed_history[track_id].pop(0)  # Hız geçmişini kısıtlı tut

                # Ortalama hızı hesapla
                avg_speed = np.mean(speed_history[track_id])

                # Yönü hesapla
                delta_x = cx - prev_cx
                delta_y = cy - prev_cy
                direction = ""

                if abs(delta_x) > abs(delta_y):  # Yatay hareket daha fazla
                    if delta_x > 0:
                        direction = "Right"
                    else:
                        direction = "Left"
                else:  # Dikey hareket daha fazla
                    if delta_y > 0:
                        direction = "Down"
                    else:
                        direction = "Up"

                # Hız ve yön bilgisini göster
                text = f"ID:{track_id} CAR, Speed: {avg_speed:.2f} px/s, Dir: {direction}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Kırmızı renk (0,0,255)

                # Ok çizimi: önceki pozisyondan şu anki pozisyona
                cv2.arrowedLine(frame, (prev_cx, prev_cy), (cx, cy), (0, 0, 255), 2, tipLength=0.5)  # Kırmızı ok

            # Şu anki pozisyonu bir sonraki kare için kaydet
            previous_positions[track_id] = (cx, cy)

            # Nesnenin izini kaydet ve çiz
            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 15:
                track.pop(0)

            points = np.hstack(track).astype("int32").reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

    # Çerçeveyi göster
    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




