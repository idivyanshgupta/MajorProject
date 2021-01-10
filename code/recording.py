import numpy as np
import cv2
import time

def record():
    
    PERIOD_OF_TIME = 10

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../recordings/output.avi',fourcc, 20.0, (640,480))
    start = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            

            out.write(frame)

            cv2.imshow('frame',frame)
            if time.time() > start + PERIOD_OF_TIME : break
        
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()