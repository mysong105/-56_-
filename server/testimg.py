import cv2
import os

#파일 위치 변경해야해요
srcdir = './images/test'

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('./haarcascade_eye.xml')

dirs = os.listdir(srcdir)
imgNum  = 0

for d in dirs:
    image = os.path.join(srcdir,d)
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
        cropped = img[y +5:y + h-10, x+5 :x + w - 10]

        # 이미지를 저장
        print("%s : 처리중" %imgNum)
        cv2.resize(cropped,dsize=(400,400))
        cv2.imwrite(srcdir+ str(imgNum) + ".jpg", cropped)
        imgNum += 1

