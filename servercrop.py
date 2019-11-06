from PIL import Image
import os
import cv2
import os


imgPath = "../d.jpg"  #image full path to crop

face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')  # give full path 
   
img = cv2.imread(imgPath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3,5)
for (x,y,w,h) in faces:
    cropped = img[y +5:y + h-10, x+5 :x + w - 10]
    cv2.resize(cropped,dsize=(400,400))
    cv2.imwrite(imgPath, cropped)
