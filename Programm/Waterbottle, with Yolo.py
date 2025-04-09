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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

                # --- ROI mit Padding ausschneiden ---
                padding = 5
                x1_p, y1_p = x1 + padding, y1 + padding
                x2_p, y2_p = x2 - padding, y2 - padding
                roi = frame[y1_p:y2_p, x1_p:x2_p]

                # --- CLAHE + Glättung ---
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(roi_gray)
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

                # --- Threshold + Canny + Morphologie ---
                _, thresh = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                edges = cv2.Canny(thresh, 50, 120)
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # --- Konturen filtern ---
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500 and len(cnt) >= 50]

                if valid_contours:
                    # Größte Kontur (höchstwahrscheinlich äußere Flasche)
                    largest = max(valid_contours, key=cv2.contourArea)

                    # --- Prüfe auf passende Flaschenform (hoch, schmal) ---
                    x_c, y_c, w_c, h_c = cv2.boundingRect(largest)
                    aspect_ratio = h_c / w_c
                    if aspect_ratio > 1.5:  # typisch für stehende Flasche

                        # --- Kontur glätten (optional) ---
                        epsilon = 0.01 * cv2.arcLength(largest, True)
                        approx = cv2.approxPolyDP(largest, epsilon, True)

                        # --- Koordinaten ins Bild zurück verschieben ---
                        approx_shifted = approx + [x1_p, y1_p]

                        # --- Stabilisierung mit vorheriger Kontur ---
                        alpha = 0.6
                        if previous_contour is not None and previous_contour.shape == approx_shifted.shape:
                            smoothed = alpha * approx_shifted.astype(np.float32) + \
                                       (1 - alpha) * previous_contour.astype(np.float32)
                            smoothed = smoothed.astype(np.int32)
                            cv2.drawContours(frame, [smoothed], -1, (0, 255, 0), 2)
                            previous_contour = smoothed
                        else:
                            cv2.drawContours(frame, [approx_shifted], -1, (0, 255, 0), 2)
                            previous_contour = approx_shifted

    # --- Bild anzeigen ---
    cv2.imshow("Flaschenerkennung + Außenkontur (clean)", frame)

    # --- Beenden mit 'q' ---
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
