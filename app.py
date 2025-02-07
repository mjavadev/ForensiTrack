from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import os
import sqlite3
import face_recognition
import cv2
import numpy as np
from detect_face_in_video import detect_face_in_video
import requests
import google.generativeai as genai
import logging
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from io import BytesIO
from flask_session import Session  # To manage sessions

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Configure Gemini API (if using Gemini model)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Hugging Face FLUX API details
HF_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Set up Hugging Face InferenceClient
client = InferenceClient(provider="fal-ai", api_key=HF_API_KEY)


app = Flask(__name__, static_folder='static')

app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback_default_key")
# Session Configuration (optional)
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data in files
Session(app)

# Configure upload folders
UPLOAD_FOLDER = 'static/images'
VIDEO_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

# Ensure upload folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Store the cumulative feedback
feedback_history = []

# Set up logging
logging.basicConfig(level=logging.INFO)


# Database connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    with get_db_connection() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
        conn.commit()

init_db()

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('index'))
    flash("Your session has expired. Please log in again.", "warning")
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for('login'))

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully", "success")
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if not username or not email or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for('signup'))

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            conn.close()
            flash("Your account has been created! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or Email already exists!", "danger")
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/index')
def index():
    if 'user' not in session:
        flash("Your session has expired. Please log in again.", "warning")
        return redirect(url_for('login'))
    return render_template('index.html') # Redirect to login if not authenticated

@app.route('/generate-sketch')
def generate_sketch():
    return render_template('generate-sketch.html')

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

@app.route('/upload-ai', methods=['POST'])
def upload_ai():
    sketch_path = request.form.get('sketch')  # Get the AI-generated image path

    if not sketch_path:
        return redirect(url_for('index'))

    # Extract filename
    filename = os.path.basename(sketch_path)

    # Move the generated image to the uploads folder for consistency
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.rename(sketch_path.lstrip('/'), saved_path)  # Move the file

    # Match the sketch with the database
    best_match, best_match_distance = match_sketch_to_db(saved_path)

    if best_match:
        return redirect(url_for('match', sketch=filename, match=best_match, distance=best_match_distance))
    else:
        return redirect(url_for('match', sketch=filename, match=None, distance=None))

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

#AI SKETCH
def refine_prompt(user_input):
    system_message = (
        "You are a highly skilled prompt engineering assistant for forensic sketches. "
        "Your task is to refine the user's input into a detailed, realistic police sketch description, "
        "including facial features, hair, clothing, and background. "
        "Use the following template for the description:\n"
        "Subject: [Gender, Age, Ethnicity, Height, Weight]\n"
        "Face: [Face shape, Cheeks, Eyes, Eyebrows, Nose, Mouth, Scars]\n"
        "Hair: [Length, Texture, Color, Style]\n"
        "Clothing: [Type, Color, Details]\n"
        "Background: [Setting, Lighting, Details]\n"
        "Ensure the description is as detailed and structured as possible."
    )
    prompt = f"{system_message}\n\nUser Input: {user_input}\n\nRefined Prompt:"
    response = gemini_model.generate_content(prompt)
    return response.text

from io import BytesIO

def generate_image_from_prompt(prompt, seed):
    logging.info(f"Starting image generation process with seed: {seed}")
    
    payload = {
        "inputs": prompt,
        "parameters": {"seed": seed}  # Ensure stable outputs using the same seed
    }
    
    MAX_RETRIES = 3
    retry_wait_time = 10  # Initial wait time (seconds)

    for attempt in range(MAX_RETRIES):
        logging.info(f"Attempt {attempt + 1} of {MAX_RETRIES}...")
        try:
            # Make the request to generate an image with a fixed seed
            response = client.text_to_image(
                prompt, model="black-forest-labs/FLUX.1-dev", seed=seed
            )
            
            # Convert the image to bytes
            image_bytes = BytesIO()
            response.save(image_bytes, format='PNG')  # Saving image as PNG
            image_bytes.seek(0)  # Reset pointer to the beginning of the byte stream
            
            # Save the image to a file
            image_path = f"static/generated_image_{seed}.png"  # Unique filename per seed
            with open(image_path, "wb") as f:
                f.write(image_bytes.read())  # Write the image bytes to the file

            logging.info("Image generated successfully.")
            return image_path
        
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            break

    logging.error("Failed to generate image after retries.")
    return None


def refine_prompt_with_feedback(original_prompt, user_feedback):
    """
    Enhances the existing prompt by applying user feedback while keeping the original structure.
    """
    if user_feedback:
        feedback_prompt = f"""
        Keep the overall structure and details of the original prompt but adjust only the necessary parts 
        based on the user feedback.

        Original Prompt:
        {original_prompt}

        User Feedback:
        {user_feedback}

        Improved Prompt:
        """
        return gemini_model.generate_content(feedback_prompt).text.strip()
    
    return original_prompt  # If no feedback, return the same prompt



import random

@app.route('/generate-image', methods=['POST'])
def generate_image():
    user_input = request.json.get('input')
    refined_prompt = refine_prompt(user_input)
    logging.info(f"Initial refined prompt: {refined_prompt}")
    # Generate a fixed seed to ensure image consistency on regeneration
    seed = random.randint(0, 999999)

    image_path = generate_image_from_prompt(refined_prompt, seed)

    if image_path:
        return jsonify({
            "image_url": f"/{image_path}",
            "refined_prompt": refined_prompt,
            "seed": seed  # Store seed for later use
        })
    
    return jsonify({"error": "Failed to generate image"}), 500


@app.route('/regenerate-image', methods=['POST'])
def regenerate_image():
    data = request.json
    logging.info(f"Received for regeneration - original refined_prompt: {data.get('refined_prompt')}, feedback: {data.get('feedback')}")
    
    # Update feedback history with the new feedback
    feedback_history.append(data.get('feedback', ''))
    refined_prompt = data.get('refined_prompt')
    
    # Combine all previous feedback and apply it to the current prompt
    for feedback in feedback_history:
        refined_prompt = refine_prompt_with_feedback(refined_prompt, feedback)

    logging.info(f"Regeneration refined prompt: {refined_prompt}")
    seed = data.get("seed")  # Retrieve the same seed
    logging.info(f"Received seed: {seed}")

    if not seed:
        seed = random.randint(0, 999999)  # Fallback seed if missing

    image_path = generate_image_from_prompt(refined_prompt, seed)

    if image_path:
        return jsonify({
            "image_url": f"/{image_path}",
            "refined_prompt": refined_prompt,
            "seed": seed  # Keep passing the same seed for consistency
        })
    
    return jsonify({"error": "Failed to regenerate image"}), 500

@app.route('/view-image')
def view_image():
    image_url = request.args.get('image_url')
    title = request.args.get('title')
    return render_template('view_image.html', image_url=image_url, title=title)

if __name__ == '__main__':
    app.run(debug=True)
