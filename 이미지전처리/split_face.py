import cv2
import os

#파일 위치 변경해야해요
srcdir = "C:\\Users\\jiki\\Documents\\deeplearning\\split_src"
desdir = "C:\\Users\\jiki\\Documents\\deeplearning\\split_dest"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('haarcascade_eye.xml')

dirs = os.listdir(srcdir)
imgNum  = 0

for d in dirs:
    image = os.path.join(srcdir,d)
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
        cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        # 이미지를 저장
        print("%s : 처리중" %imgNum)
        cv2.imwrite("./split_dest/" + str(imgNum) + ".jpg", cropped)
        imgNum += 1

cv2.waitKey(0)
cv2.destroyAllWindows()