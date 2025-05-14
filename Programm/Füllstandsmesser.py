
# Autor:            Bachmann Tim
# Skript Name:    Füllstandsmesser
# Github Link: https://github.com/TimPatrickProgramming/Bildverarbeitung_1_Projektarbeit.git
# Beschreibung:
# Dieses Skript erkennt den Füllstand einer Durchsichtigen Flasche
# und gibt diesen Abhängig von der Höhe in % sowie als eine Angabe in L aus.



# ----- Einbindung der Bibliotheken -----
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO



# ----- Erstellen des GUIs -----

# Dropdown-Optionen für Referenzvolumen der zu messenden Wasserflasche
volumen_optionen = {
    "0.33 L": 0.33,
    "0.5 L": 0.5,
    "1.0 L": 1.0,
    "1.25 L": 1.25,
    "1.5 L": 1.5
}

# Name und Grösse des Fensters festlegen
root = tk.Tk()
root.title("Flaschenanalyse")
root.geometry("1200x720")

# Erstellen eines linken Feldes zum Einschreiben der Daten
left_frame = tk.Frame(root, width=250, bg="white")
left_frame.pack(side="left", fill="y")

# Erstellen eines rechten Feldes zum Anzeigen des Livefeeds
right_frame = tk.Frame(root, bg="black")
right_frame.pack(side="right", expand=True, fill="both")

# Erstellen des Dropdown Menus zur Auswahl des Referenzvolumens
selected_volume = tk.StringVar(root)
selected_volume.set("1.5 L")
tk.Label(left_frame, text="Flaschengröße auswählen:", bg="white").pack(pady=(20, 5))
dropdown = tk.OptionMenu(left_frame, selected_volume, *volumen_optionen.keys())
dropdown.pack(pady=(0, 20))

# Erstellen des Labels zur Anzeige der Messwerte
info_label = tk.Label(left_frame, text="Werte", justify="left", bg="white", font=("Courier", 12))
info_label.pack(padx=10, anchor="nw")

# Erstellen des Labels zur Anzeige des Livefeeds
video_label = tk.Label(right_frame)
video_label.pack(fill="both", expand=True)

# Initialisieren des Yolo Modells und des Kamera inputs
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Initialisieren der Counter welche die Updaterate der Messwerte bestimmen
previous_contour = None
stable_counter = 0
max_missing_frames = 15
frame_count = 0
update_interval = 5
last_percent = None
last_volume = None
last_y_water = None



# ----- Funktionen deklaration -----

# Funktion zum berechnen des Volumens
def berechne_volumenbereich(smoothed_contour, y_start, y_end):
    x, y, w, h = cv2.boundingRect(smoothed_contour)     # bestimmen des kleinsten Rechtecks der Flaschenkontur
    radii = []
    # Berechnen der Radien der Flaschenkontur
    for i in range(y_start, y_end):
        mask = np.zeros((y + h + 10, x + w + 10), dtype=np.uint8)
        cv2.drawContours(mask, [smoothed_contour], -1, 255, -1)
        row = mask[i, :]
        cols = np.where(row == 255)[0]
        # Falls ein Objekt gefunden wurde Radius berechnen und der Liste hinzufügen
        if len(cols) > 1:
            r = (cols[-1] - cols[0]) / 2
            radii.append(r)
    # Falls kein Radius gefunden Volumen 0L zurückgeben, sowie Faktor 1 um division durch 0 zu verhindern
    if not radii:
        return 0.0, 1.0
    # Umwandeln des Radien array in Numpy array und brechnen des Volumens der Flasche in pixel
    radii = np.array(radii)
    dy = 1
    vol_px3 = np.sum(np.pi * (radii ** 2) * dy)
    return vol_px3, len(radii)      # Rückgabe des Flaschenvolumens in Pixel und anzahl einbezogener schichten


# Hauptfunktion welche alle berechnungen ausführt welche in einem Frame gemacht werden
def update_frame():
    # Globale Variablen zum Anzeigen der Kontur
    global previous_contour, stable_counter, frame_count
    global last_percent, last_volume, last_y_water

    # Einlesen eines Frames mit anschliessender Kontrolle
    ret, frame = cap.read()
    if not ret:
        return

    # Referenzvolumen updaten
    selected_liter = volumen_optionen[selected_volume.get()]
    
    # Flasche mit Yolo erkennen um erste ROI zu definieren
    results = model(frame)
    current_contour = None

    # Alle erkannte Objekte nach Flaschen absuchen
    for r in results:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "bottle":
                # Koordinaten der Flasche 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                padding = 5
                x1_p, y1_p = x1 + padding, y1 + padding
                x2_p, y2_p = x2 - padding, y2 - padding
                # Erstes ROI ausschneiden um die Position der Flasche Grob zu bestimmen, ROI nachbearbeiten
                roi = frame[y1_p:y2_p, x1_p:x2_p]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(roi_gray)
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                
                # Kanten der Flasche mithilfe von Sobel detektieren
                sobel = np.uint8(np.clip(np.hypot(
                    cv2.Sobel(blurred, cv2.CV_64F, 1, 0),
                    cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
                ) * 2.0, 0, 255))
                
                # Binärbild sowie Kontur extrahieren
                _, binary = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Grösste Gültige Kontur suchen und Abspeichern, weiteres Eingrenzen der ROI
                valid = [c for c in contours if cv2.contourArea(c) > 300]
                if valid:
                    largest = max(valid, key=cv2.contourArea)
                    current_contour = largest + [x1_p, y1_p]

    # Stabilität der Kontur prüfen, vorgehen gegen Flackern
    if current_contour is not None:
        if previous_contour is not None and previous_contour.shape == current_contour.shape:
            # Falls sich die Kontur nicht gross verändert und ähnlich bleibt wird diese als Stabil bewertet
            if np.mean(np.abs(current_contour.astype(np.float32) - previous_contour.astype(np.float32))) < 10:
                previous_contour = current_contour
                stable_counter = 0
        else:
            previous_contour = current_contour
            stable_counter = 0
    # Prüfen ob die Kontur noch vorhanden ist
    else:
        stable_counter += 1
        if stable_counter > max_missing_frames:
            previous_contour = None

    # Standardanzeige für das GUI definieren
    fuellstand_text = "Fuellstand: ---\nFuellvol.: --- L\nGesamtvol.: --- L"
    frame_count += 1

    if previous_contour is not None:
        # Glätten der Kontur um Flasche genauer erkennen zu können
        epsilon = 0.01 * cv2.arcLength(previous_contour, True)
        smoothed = cv2.approxPolyDP(previous_contour, epsilon, True)
        cv2.drawContours(frame, [smoothed], -1, (0, 255, 255), 1)

        # Maske erstellen um Deckel auszuschliessen, dies ist nötig um zu verhindern das deckel als füllstand erkannt wird, verringert flackern
        x, y, w, h = cv2.boundingRect(smoothed)
        fixed_deckel_h = int(0.08 * h)
        mask = np.zeros_like(frame[:, :, 0])
        cv2.drawContours(mask, [smoothed], -1, 255, -1)
        deckel_mask = np.zeros_like(mask)
        cv2.rectangle(deckel_mask, (x, y), (x + w, y + fixed_deckel_h), 255, -1)
        mask_no_deckel = cv2.subtract(mask, deckel_mask)
        
        # Neue ROI ohne flaschendeckel für weitere Analyse
        masked = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask_no_deckel)
        strip = masked[y:y + h, x + w // 2 - 2:x + w // 2 + 2]
        profile = np.mean(strip, axis=1)
        diff = np.diff(profile)

        if len(diff) > 0:
            # Suchen der Wasseroberfläche / Stelle mit grösstem Helligkeitsunterschied
            idx = np.argmax(diff)
            y_water = y + idx
            percent = (h - idx) / h     # Prozentualer Höhenabhängiger Füllstand berechnen
            
            # Berechnen des Volumens der Gesamtflasche und des Flascheninhaltes
            vol_fuell_px3, _ = berechne_volumenbereich(smoothed, y + idx, y + h)
            vol_gesamt_px3, _ = berechne_volumenbereich(smoothed, y, y + h)
            voll_liter = (vol_fuell_px3 / vol_gesamt_px3) * selected_liter

            # Frame Update nur bei grösseren veränderungen, eindämmen von Flackern
            if last_y_water is None or abs(y_water - last_y_water) > 2:
                last_y_water = y_water

            last_percent = percent
            last_volume = voll_liter

        # Einzeichnen der Wasserlinie, falls definiert
        if last_y_water:
            cv2.line(frame, (x, last_y_water), (x + w, last_y_water), (0, 0, 255), 2)

        # Aktualisieren des Textes im GUI
        if last_percent is not None and last_volume is not None:
            fuellstand_text = f"Fuellstand: {int(last_percent * 100)}%\nFuellvol.: {last_volume:.2f} L\nGesamtvol.: {selected_liter:.2f} L"
    info_label.config(text=fuellstand_text)

    # Livefeed an Fenstergrösse anpassen
    target_w = video_label.winfo_width()
    target_h = video_label.winfo_height()
    if target_w < 100 or target_h < 100:
        target_w, target_h = 640, 480

    frame_resized = cv2.resize(frame, (target_w, target_h))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    #Funktion alle 10ms widerholen
    root.after(10, update_frame)

# Aufrufen der Funktionen
update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
