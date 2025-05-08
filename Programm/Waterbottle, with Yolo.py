from ultralytics import YOLO
import cv2
import numpy as np

threshold_val = 100
sobel_gain = 2.0
min_area = 300

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

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

                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(roi_gray)
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

                sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                sobel = cv2.magnitude(sobelx, sobely)
                sobel = np.uint8(np.clip(sobel * sobel_gain, 0, 255))

                _, binary = cv2.threshold(sobel, threshold_val, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

                if valid:
                    largest = max(valid, key=cv2.contourArea)
                    current_contour = largest + [x1_p, y1_p]

    # --- Stabilisierung ---
    if current_contour is not None:
        if previous_contour is not None and previous_contour.shape == current_contour.shape:
            diff = np.mean(np.abs(current_contour.astype(np.float32) - previous_contour.astype(np.float32)))
            if diff < 10:
                previous_contour = current_contour
                stable_counter = 0
        else:
            previous_contour = current_contour
            stable_counter = 0
    else:
        stable_counter += 1
        if stable_counter > max_missing_frames:
            previous_contour = None

    # --- Analyse & Anzeige ---
    if previous_contour is not None:
        cv2.drawContours(frame, [previous_contour], -1, (0, 255, 0), 2)

        # Maske aus Flasche
        mask = np.zeros_like(frame[:, :, 0])
        cv2.drawContours(mask, [previous_contour], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(previous_contour)

        # Deckelregion definieren
        top_region = mask[y:y + int(0.25 * h), x:x + w]
        deckel_contours, _ = cv2.findContours(top_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        deckel_mask = np.zeros_like(mask)
        for cnt in deckel_contours:
            dx, dy, dw, dh = cv2.boundingRect(cnt)
            aspect = dw / max(dh, 1)
            if 0.8 < aspect < 2.5 and dh < h * 0.15:
                cv2.drawContours(deckel_mask[y:y + int(0.25 * h), x:x + w], [cnt], -1, 255, -1)

        # Deckel aus der Flaschenmaske entfernen
        mask_no_deckel = cv2.subtract(mask, deckel_mask)
        masked = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask_no_deckel)

        # Wasserstand ermitteln
        profile_strip = masked[y:y + h, x + w // 2 - 2:x + w // 2 + 2]
        vertical_profile = np.mean(profile_strip, axis=1)

        diff = np.diff(vertical_profile)
        if len(diff) > 0:
            water_idx = np.argmax(diff)
            water_y = y + water_idx

            # Linie & Prozent
            cv2.line(frame, (x, water_y), (x + w, water_y), (0, 0, 255), 2)
            f_percent = int(100 * (h - water_idx) / h)
            cv2.putText(frame, f"Fuellstand: {f_percent}%", (x + w + 10, water_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Ausgabe skalieren (2x)
    scale = 2.0
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Flaschenanalyse (Deckel erkannt)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
