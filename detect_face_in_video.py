import face_recognition
import cv2
import numpy as np

# Load the matched image
matched_image_path = "database/vkimg.jpg"  # Replace with the best match from Step 3
matched_image = face_recognition.load_image_file(matched_image_path)
matched_encoding = face_recognition.face_encodings(matched_image)[0]

# Open the video
video_path = "video.mp4"
video_capture = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with the matched image
        distance = np.linalg.norm(matched_encoding - face_encoding)

        # If the distance is below a threshold, it's a match
        if distance < 0.6:  # You can adjust this threshold
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Match", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Video", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close windows
video_capture.release()
cv2.destroyAllWindows()