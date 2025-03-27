# --- Import Librarys --- #

import os
import numpy as np
import cv2


# --- Camera Setup --- #

DEVICE_ID = 0
# open the Camera
videoBackend = cv2.CAP_DSHOW
cap = cv2.VideoCapture(DEVICE_ID, videoBackend)

# # --- Kameraeinstellungen --- #
# # Achtung: Werte hängen von der Kamera ab – evtl. musst du testen
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # Manuelle Belichtung
# cap.set(cv2.CAP_PROP_EXPOSURE, -6)          # Belichtungszeit (je kleiner, desto dunkler)
# cap.set(cv2.CAP_PROP_GAIN, 0)               # Verstärkung (ISO) – höher = heller, aber mehr Rauschen

# --- Function declarations --- #



# check if Camera was opened succesfull
if not cap.isOpened():
        print('ERROR: could not open webcam')


        
# --- Main Loop --- #    
   
while(True):
    # read a single frame from the camera
    ret, frame = cap.read()
    
    # check if frame was captured
    if not ret:
        print('ERROR: could not read data from webcam')
        break
    
    # Preparation of the Image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # Generate Grayscale Picture
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)         # Smoothing out the image to reduce noise
    edges = cv2.Canny(blurred, 50, 250)                 # Find edges in the image
    
    # Find all contures
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the most logical conture for the Water bottle
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 1500:  # filter against smaler artifacts
            # draw conture
            cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 2)
    
    
    # show the live Image
    cv2.imshow("Press 'q' to quit.", frame)
    
    # Check for User Input to end or continue the Live feed
    ch = cv2.waitKey(20)
    if ch==ord('q'):
        break
    if ch==ord('0'):
        cap.set(cv2.CAP_PROP_SETTINGS,0)
   
# Close the Window and Camera
cap.release()
cv2.destroyAllWindows()