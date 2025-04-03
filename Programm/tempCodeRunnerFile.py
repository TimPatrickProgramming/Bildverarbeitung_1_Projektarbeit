from ultralytics import YOLO
import cv2
import numpy as np

# --- YOLOv8 Modell laden ---
model = YOLO("yolov8n.pt")

# --- Webcam starten ---
cap = cv2.VideoCapture(0)

# --- Letzte Kontur zur Stabilisierung speichern ---
previous_contour = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "bottle":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Optional: Bounding Box zeichnen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

                # --- ROI mit Innen-Padding ausschneiden ---
                padding = 5
                x1_p, y1_p = x1 + padding, y1 + padding
                x2_p, y2_p = x2 - padding, y2 - padding
                roi = frame[y1_p:y2_p, x1_p:x2_p]

                # --- CLAHE-Kontrastverstärkung ---
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(roi_gray)

                # --- Glättung + Thresholding ---
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 15, 2)

                # --- Canny-Kanten + Morphologie ---
                edges = cv2.Canny(thresh, 50, 120)
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # --- Konturen extrahieren ---
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 500:
                        # Kontur auf das Originalbild zurückverschieben
                        largest_shifted = largest + [x1_p, y1_p]

                        # --- Kontur glätten (Tracking-Effekt) ---
                        alpha = 0.6
                        if previous_contour is not None and previous_contour.shape == largest_shifted.shape:
                            smoothed = alpha * largest_shifted.astype(np.float32) + \
                                    (1 - alpha) * previous_contour.astype(np.float32)
                            smoothed = smoothed.astype(np.int32)
                            cv2.drawContours(frame, [smoothed], -1, (0, 255, 0), 2)
                            previous_contour = smoothed
                        else:
                            cv2.drawContours(frame, [largest_shifted], -1, (0, 255, 0), 2)
                            previous_contour = largest_shifted


    # --- Livebild anzeigen ---
    cv2.imshow("Flaschenerkennung + Kontur (stabilisiert)", frame)

    # --- Beenden mit 'q' ---
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
