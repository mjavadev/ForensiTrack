from flask import Flask, render_template, request, redirect, url_for
import os
import face_recognition
import cv2
import numpy as np
from detect_face_in_video import detect_face_in_video


app = Flask(__name__, static_folder='static')

# Configure upload folders
UPLOAD_FOLDER = 'static/images'
VIDEO_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

# Ensure upload folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Handle sketch upload and matching
@app.route('/upload', methods=['POST'])
def upload():
    if 'sketch' not in request.files:
        return redirect(url_for('index'))

    # Save the uploaded sketch
    sketch = request.files['sketch']
    sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], sketch.filename)
    sketch.save(sketch_path)

    # Match the sketch to the database
    best_match, best_match_distance = match_sketch_to_db(sketch_path)

    if best_match:
        return redirect(url_for('match', sketch=sketch.filename, match=best_match, distance=best_match_distance))
    else:
        return redirect(url_for('match', sketch=sketch.filename, match=None, distance=None))

# Match result page
@app.route('/match')
def match():
    sketch = request.args.get('sketch')
    match = request.args.get('match')
    distance = request.args.get('distance')
    return render_template('match.html', sketch=sketch, match=match, distance=distance)

# Video detection page
@app.route('/video_detection')
def video_detection():
    match = request.args.get('match')
    return render_template('detect.html', match=match)

# Handle video upload and face detection
# Handle video upload and face detection
@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    # Save the uploaded video
    video = request.files['video']
    video_path = os.path.join(app.config['VIDEO_FOLDER'], video.filename)
    video.save(video_path)

    # Get the matched image path
    matched_image_path = os.path.join('static', 'database', request.form['match'])

    if not os.path.exists(matched_image_path):
        return "Matched image not found in the database.", 404

    # Detect the matched face in the video
    detected, screenshot_path = detect_face_in_video(video_path, matched_image_path)

    print(f"Detection result: {detected}, Screenshot path: {screenshot_path}")  # Debug print

    detected_str = 'True' if detected else 'False'

    if detected:
        return redirect(url_for('detect_result', detected=detected_str, screenshot=screenshot_path))
    else:
        return redirect(url_for('detect_result', detected=detected_str))


# Video detection result page
# Video detection result page
@app.route('/detect_result')
def detect_result():
    detected = request.args.get('detected')
    screenshot = request.args.get('screenshot')

    # Replace backslashes with forward slashes
    screenshot = screenshot.replace("\\", "/")

    print(f"Detected: {detected}, Screenshot: {screenshot}")  # Debug print

    if detected == 'True':
        return render_template('detect_result.html', detected=True, screenshot='images/screenshot.jpg')
    else:
        return render_template('detect_result.html', detected=False)




# Match sketch to database
def match_sketch_to_db(sketch_path):
    sketch_image = face_recognition.load_image_file(sketch_path)
    sketch_encoding = face_recognition.face_encodings(sketch_image)

    if not sketch_encoding:
        return None, None

    sketch_encoding = sketch_encoding[0]
    database_folder = 'static/database'
    best_match = None
    best_match_distance = float('inf')

    for image_name in os.listdir(database_folder):
        image_path = os.path.join(database_folder, image_name)
        db_image = face_recognition.load_image_file(image_path)
        db_encoding = face_recognition.face_encodings(db_image)

        if db_encoding:
            db_encoding = db_encoding[0]
            distance = np.linalg.norm(sketch_encoding - db_encoding)
            if distance < best_match_distance:
                best_match_distance = distance
                best_match = image_name

    return best_match, best_match_distance


if __name__ == '__main__':
    app.run(debug=True)
