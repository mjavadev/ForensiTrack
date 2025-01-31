import face_recognition
import cv2
import sys
import os
import numpy as np

def detect_face_in_video(video_path, matched_image_path):
    # Debug: Check paths for input files
    print(f"Video Path: {video_path}")
    print(f"Matched Image Path: {matched_image_path}")
    
    # Load the matched image and extract encoding
    matched_image = face_recognition.load_image_file(matched_image_path)
    matched_encoding = face_recognition.face_encodings(matched_image)

    # Debug: Check if encoding is extracted
    if not matched_encoding:
        print("No face encoding found for the matched image.")
        return False, None
    
    matched_encoding = matched_encoding[0]
    print("Matched encoding extracted successfully.")

    # Open the video capture
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Unable to open video.")
        return False, None
    
    detected = False
    screenshot_path = None
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or failed to capture frame.")
            break
        
        # Convert frame to RGB (for face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print(f"Detected {len(face_locations)} faces in this frame.")  # Debug print

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare face with the matched encoding
            distance = np.linalg.norm(matched_encoding - face_encoding)
            print(f"Distance to matched face: {distance}")  # Debug print

            # If the face matches
            if distance < 0.6:  # You can adjust the threshold based on your needs
                detected = True
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Detected", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Save the screenshot
                screenshot_filename = 'screenshot.jpg'
                screenshot_path = os.path.join('static', 'images', screenshot_filename)
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved at: {screenshot_path}")  # Debug print
                break
        
        if detected:
            break
    
    video_capture.release()
    return detected, screenshot_path

# Test run when running the script directly (optional for debugging)
if __name__ == "__main__":
    video_path = sys.argv[1]  # Video path passed as argument
    matched_image_path = sys.argv[2]  # Matched image path passed as argument
    
    detected, screenshot_filename = detect_face_in_video(video_path, matched_image_path)

    if detected:
        print(f"Detected {screenshot_filename}")
    else:
        print("No face detected.")
