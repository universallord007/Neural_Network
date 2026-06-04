import cv2
import os

face_detector = cv2.CascadeClassifier(r"C:\Users\Dipankar\AppData\Local\Programs\Python\Python312\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")


img = cv2.imread("faces2.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# draw boxes around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

print("Faces found:", len(faces))

# show result
cv2.imshow("Face Detection", img)
cv2.waitKey(0)