import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# load and resize image bigger
img = cv2.imread("face.jpg")
img = cv2.resize(img, (1280, 720))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp_face.FaceDetection(min_detection_confidence=0.2) as detector:
    results = detector.process(rgb)

    if results.detections:
        print("Faces found:", len(results.detections))
        for face in results.detections:
            mp_draw.draw_detection(img, face)
    else:
        print("No faces found")

cv2.imshow("MediaPipe Face Detection", img)
cv2.waitKey(0)