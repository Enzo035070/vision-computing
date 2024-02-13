# import cv2 
# import time
# vid = cv2.VideoCapture(0) 
# starttime = time.time()
# while(True): 


# 	ret, frame = vid.read() 
# 	cv2.imshow('window', frame)
# 	# cv2.imwrite(f'/Users/enzo/projects/CLub/images/saved{i}.png', frame) 
	
# 	if cv2.waitKey(1) & 0xFF == ord('q'): 
# 		break
# endtime = time.time()
# framerate = 100/(endtime - starttime)
# print(framerate)
# vid.release() 
# cv2.destroyAllWindows() 

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break

cap.release()
cv2.destroyAllWindows()
