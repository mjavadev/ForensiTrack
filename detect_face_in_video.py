import face_recognition
import cv2
import numpy as np
import sys

# Get arguments from subprocess call
matched_image_path = sys.argv[1]
video_path = sys.argv[2]

# Load the matched image
matched_image = face_recognition.load_image_file(matched_image_path)
matched_encoding = face_recognition.face_encodings(matched_image)[0]

# Open the video
video_capture = cv2.VideoCapture(video_path)
detected = False

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distance = np.linalg.norm(matched_encoding - face_encoding)
        if distance < 0.6:
            detected = True
            screenshot_path = "static/images/screenshot.jpg"
            cv2.imwrite(screenshot_path, frame)
            break

    if detected:
        break

video_capture.release()
cv2.destroyAllWindows()
