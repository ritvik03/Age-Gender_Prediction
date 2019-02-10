import numpy as np
import cv2 as cv
import sys
face_cascade = cv.CascadeClassifier(sys.argv[2]+'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(sys.argv[2]+'haarcascade_eye.xml')
img = cv.imread(sys.argv[1])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#cv.imshow('img',img)
cv.imwrite(sys.argv[2]+'_face_eyes.jpg',img)
print("Face and Eye detection results saved")
#cv.waitKey(0)
cv.destroyAllWindows()
