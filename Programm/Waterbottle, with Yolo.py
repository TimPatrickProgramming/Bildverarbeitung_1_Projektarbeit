import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Dropdown-Optionen
volumen_optionen = {
    "0.33 L": 0.33,
    "0.5 L": 0.5,
    "1.0 L": 1.0,
    "1.25 L": 1.25,
    "1.5 L": 1.5
}

# TK GUI Setup
root = tk.Tk()
root.title("Flaschenanalyse")
root.geometry("1200x720")

left_frame = tk.Frame(root, width=250, bg="white")
left_frame.pack(side="left", fill="y")

right_frame = tk.Frame(root, bg="black")
right_frame.pack(side="right", expand=True, fill="both")

selected_volume = tk.StringVar(root)
selected_volume.set("1.5 L")
tk.Label(left_frame, text="Flaschengröße auswählen:", bg="white").pack(pady=(20, 5))
dropdown = tk.OptionMenu(left_frame, selected_volume, *volumen_optionen.keys())
dropdown.pack(pady=(0, 20))

info_label = tk.Label(left_frame, text="Werte", justify="left", bg="white", font=("Courier", 12))
info_label.pack(padx=10, anchor="nw")

video_label = tk.Label(right_frame)
video_label.pack(fill="both", expand=True)

# YOLO Modell
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

previous_contour = None
stable_counter = 0
max_missing_frames = 15

frame_count = 0
update_interval = 5
last_percent = None
last_volume = None
last_y_water = None

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

def update_frame():
    global previous_contour, stable_counter, frame_count
    global last_percent, last_volume, last_y_water

    ret, frame = cap.read()
    if not ret:
        return

    selected_liter = volumen_optionen[selected_volume.get()]
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

    fuellstand_text = "Fuellstand: ---\nFuellvol.: --- L\nGesamtvol.: --- L"
    frame_count += 1

    if previous_contour is not None:
        epsilon = 0.01 * cv2.arcLength(previous_contour, True)
        smoothed = cv2.approxPolyDP(previous_contour, epsilon, True)
        cv2.drawContours(frame, [smoothed], -1, (0, 255, 255), 1)

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
            voll_liter = (vol_fuell_px3 / vol_gesamt_px3) * selected_liter

            if last_y_water is None or abs(y_water - last_y_water) > 2:
                last_y_water = y_water

            last_percent = percent
            last_volume = voll_liter

        # Linie immer zeichnen
        if last_y_water:
            cv2.line(frame, (x, last_y_water), (x + w, last_y_water), (0, 0, 255), 2)

        if last_percent is not None and last_volume is not None:
            fuellstand_text = f"Fuellstand: {int(last_percent * 100)}%\nFuellvol.: {last_volume:.2f} L\nGesamtvol.: {selected_liter:.2f} L"

    info_label.config(text=fuellstand_text)

    # Dynamische Skalierung des Video-Fensters
    target_w = video_label.winfo_width()
    target_h = video_label.winfo_height()
    if target_w < 100 or target_h < 100:
        target_w, target_h = 640, 480

    frame_resized = cv2.resize(frame, (target_w, target_h))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Starte GUI
update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
