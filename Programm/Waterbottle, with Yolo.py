from ultralytics import YOLO
import cv2
import numpy as np

# Parameter
threshold_val = 100
sobel_gain = 2.0
min_area = 300
FLASCHE_VOL_LITER = 1.5

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

previous_contour = None
stable_counter = 0
max_missing_frames = 15

def berechne_volumen_liter_kontur(smoothed_contour, y_start, y_end, full_h_px, referenz_liter=1.5):
    x, y, w, h = cv2.boundingRect(smoothed_contour)
    radii = []
    for i in range(y_start, y_end):
        mask = np.zeros((y + h + 10, x + w + 10), dtype=np.uint8)
        cv2.drawContours(mask, [smoothed_contour], -1, 255, -1)
        row = mask[i, :]
        cols = np.where(row == 255)[0]
        if len(cols) > 1:
            r = (cols[-1] - cols[0]) / 2
            radii.append(r)
    if not radii:
        return 0.0
    radii = np.array(radii)
    dy = 1  # 1 Pixel
    volume_px3 = np.sum(np.pi * (radii ** 2) * dy)
    full_radii = np.linspace(np.max(radii), np.min(radii), full_h_px)
    full_volume_px3 = np.sum(np.pi * (full_radii ** 2) * dy)
    scale = referenz_liter / full_volume_px3
    return volume_px3 * scale

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_contour = None

    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "bottle":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                padding = 5
                x1_p, y1_p = x1 + padding, y1 + padding
                x2_p, y2_p = x2 - padding, y2 - padding
                roi = frame[y1_p:y2_p, x1_p:x2_p]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(roi_gray)
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                sobel = np.uint8(np.clip(np.hypot(
                    cv2.Sobel(blurred, cv2.CV_64F, 1, 0),
                    cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
                ) * sobel_gain, 0, 255))
                _, binary = cv2.threshold(sobel, threshold_val, 255, cv2.THRESH_BINARY)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in contours if cv2.contourArea(c) > min_area]
                if valid:
                    largest = max(valid, key=cv2.contourArea)
                    current_contour = largest + [x1_p, y1_p]

    if current_contour is not None:
        if previous_contour is not None and previous_contour.shape == current_contour.shape:
            if np.mean(np.abs(current_contour.astype(np.float32) - previous_contour.astype(np.float32))) < 10:
                previous_contour = current_contour
                stable_counter = 0
        else:
            previous_contour = current_contour
            stable_counter = 0
    else:
        stable_counter += 1
        if stable_counter > max_missing_frames:
            previous_contour = None

    if previous_contour is not None:
        epsilon = 0.01 * cv2.arcLength(previous_contour, True)
        smoothed = cv2.approxPolyDP(previous_contour, epsilon, True)
        cv2.drawContours(frame, [smoothed], -1, (0, 255, 255), 1)  # 🟡 Debuglinie

        # cv2.drawContours(frame, [previous_contour], -1, (0, 255, 0), 1)  # 🟢 Original (auskommentiert)

        x, y, w, h = cv2.boundingRect(smoothed)
        fixed_deckel_h = int(0.08 * h)
        mask = np.zeros_like(frame[:, :, 0])
        cv2.drawContours(mask, [smoothed], -1, 255, -1)
        deckel_mask = np.zeros_like(mask)
        cv2.rectangle(deckel_mask, (x, y), (x + w, y + fixed_deckel_h), 255, -1)
        mask_no_deckel = cv2.subtract(mask, deckel_mask)
        masked = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask_no_deckel)

        strip = masked[y:y + h, x + w // 2 - 2:x + w // 2 + 2]
        profile = np.mean(strip, axis=1)
        diff = np.diff(profile)
        if len(diff) > 0:
            idx = np.argmax(diff)
            y_water = y + idx
            cv2.line(frame, (x, y_water), (x + w, y_water), (0, 0, 255), 2)

            # 🔴 Prozent & Volumenanzeige oben rechts
            percent = (h - idx) / h
            voll_liter = berechne_volumen_liter_kontur(smoothed, y + idx, y + h, h, FLASCHE_VOL_LITER)
            cv2.putText(frame, f"Fuellstand: {int(percent * 100)}%", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Fuellvol.: {voll_liter:.2f} L", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Gesamtvolumen (fest normiert)
        cv2.putText(frame, f"Gesamtvol.: {FLASCHE_VOL_LITER:.2f} L", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Flaschenanalyse (Fuellvolumen)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
