# --- Import Librarys --- #

import os
import numpy as np
import cv2


# --- Variable declarations --- #

DEVICE_ID = 0



# --- Function declarations --- #


# open the Camera
videoBackend = cv2.CAP_DSHOW
cap = cv2.VideoCapture(DEVICE_ID, videoBackend)

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