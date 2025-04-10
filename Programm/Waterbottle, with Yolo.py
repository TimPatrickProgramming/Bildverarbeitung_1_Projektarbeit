from ultralytics import YOLO
import cv2
import numpy as np

# Feste Parameter
threshold_val = 100
sobel_gain = 2.0
min_area = 300

# Modell & Kamera
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Stabilitätsspeicher
previous_contour = None
stable_counter = 0
max_missing_frames = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_contour = None

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "bottle":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

                padding = 5
                x1_p, y1_p = x1 + padding, y1 + padding
                x2_p, y2_p = x2 - padding, y2 - padding
                roi = frame[y1_p:y2_p, x1_p:x2_p]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Kontrastverstärkung
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(roi_gray)
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

                # Sobel-Kanten
                sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                sobel = cv2.magnitude(sobelx, sobely)
                sobel = np.uint8(np.clip(sobel * sobel_gain, 0, 255))

                # Binärschwelle
                _, binary = cv2.threshold(sobel, threshold_val, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # Konturen finden
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

                if valid:
                    largest = max(valid, key=cv2.contourArea)
                    current_contour = largest + [x1_p, y1_p]

    # --- Kontur-Stabilisierung ---
    if current_contour is not None:
        if previous_contour is not None and previous_contour.shape == current_contour.shape:
            diff = np.mean(np.abs(current_contour.astype(np.float32) - previous_contour.astype(np.float32)))
            if diff < 10:  # Änderung ist klein → aktualisiere
                previous_contour = current_contour
                stable_counter = 0
        else:
            previous_contour = current_contour
            stable_counter = 0
    else:
        stable_counter += 1
        if stable_counter > max_missing_frames:
            previous_contour = None  # Zurücksetzen bei zu vielen fehlenden Frames

    # --- Kontur anzeigen ---
    if previous_contour is not None:
        cv2.drawContours(frame, [previous_contour], -1, (0, 255, 0), 2)

    # Anzeige
    cv2.imshow("Flaschenerkennung (stabilisiert)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
