import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from ultralytics import YOLO
import threading

# Globale Variable
FLASCHE_VOL_LITER = 1.5
laufend = False

# Volumenfunktion (wie im bisherigen Code)
def berechne_volumenbereich(smoothed_contour, y_start, y_end):
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
        return 0.0, 1.0
    radii = np.array(radii)
    dy = 1
    vol_px3 = np.sum(np.pi * (radii ** 2) * dy)
    return vol_px3, len(radii)

# Erkennungsschleife in separatem Thread
def erkennung_starten():
    global FLASCHE_VOL_LITER, laufend
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    previous_contour = None
    stable_counter = 0
    max_missing_frames = 15

    while laufend:
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
                    ) * 2.0, 0, 255))
                    _, binary = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in contours if cv2.contourArea(c) > 300]
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
                percent = (h - idx) / h
                vol_fuell_px3, _ = berechne_volumenbereich(smoothed, y + idx, y + h)
                vol_gesamt_px3, _ = berechne_volumenbereich(smoothed, y, y + h)
                voll_liter = (vol_fuell_px3 / vol_gesamt_px3) * FLASCHE_VOL_LITER

                cv2.line(frame, (x, y_water), (x + w, y_water), (0, 0, 255), 2)
                texts = [
                    f"Fuellstand: {int(percent * 100)}%",
                    f"Fuellvolumen: {voll_liter:.2f} L",
                    f"Gesamtvolumen: {FLASCHE_VOL_LITER:.2f} L"
                ]
                for i, t in enumerate(texts):
                    cv2.putText(frame, t, (20, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        frame = cv2.resize(frame, None, fx=2.0, fy=2.0)
        cv2.imshow("Flaschenanalyse (GUI)", frame)
        if cv2.waitKey(1) == ord('q'):
            laufend = False
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Fenster
def gui_start():
    def set_volumen(event):
        global FLASCHE_VOL_LITER
        auswahl = dropdown.get()
        try:
            FLASCHE_VOL_LITER = float(auswahl.split(" ")[0])
        except:
            pass

    def start():
        global laufend
        laufend = True
        t = threading.Thread(target=erkennung_starten)
        t.start()

    root = tk.Tk()
    root.title("Flaschenanalyse GUI")
    root.geometry("300x150")

    ttk.Label(root, text="Waehle Flaschenvolumen:").pack(pady=5)
    dropdown = ttk.Combobox(root, values=["0.5 L", "1.0 L", "1.5 L", "2.0 L"])
    dropdown.current(2)
    dropdown.bind("<<ComboboxSelected>>", set_volumen)
    dropdown.pack(pady=5)

    start_btn = ttk.Button(root, text="Starte Analyse", command=start)
    start_btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    gui_start()
