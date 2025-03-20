# Import Librarys

import os
import numpy as np
import cv2

# define Variables

DEVICE_ID = 0

# --- Function declarations --- #

# --- Main Loop --- #

videoBackend = cv2.CAP_DSHOW
cap = cv2.VideoCapture(DEVICE_ID, videoBackend)

if not cap.isOpened():
        print('ERROR: could not open webcam')
        
while(True):
    ret, frame = cap.read()
    if not ret:
        print('ERROR: could not read data from webcam')
        break
    
    cv2.imshow("Press 'q' to quit.", frame)
    ch = cv2.waitKey(20)
    if ch==ord('q'):
        break
    if ch==ord('0'):
        cap.set(cv2.CAP_PROP_SETTINGS,0)
   

cap.release()
cv2.destroyAllWindows()