import face_recognition
import cv2
import os
import numpy as np
import pickle
from multiprocessing import Pool, freeze_support
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Function to resize images
def resize_image(image, size=(128, 128)):
    return cv2.resize(image, size)

# Function to process a single image and extract face encoding
def process_image(image_path):
    try:
        # Load and resize the image
        image = face_recognition.load_image_file(image_path)
        image = resize_image(image)

        # Extract face encoding
        encodings = face_recognition.face_encodings(image)
        if encodings:  # If a face is found
            return image_path, encodings[0]
        else:
            return None
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None

def main():
    # Load the sketch
    sketch_path = "vk.jpg"
    sketch_image = face_recognition.load_image_file(sketch_path)
    sketch_image = resize_image(sketch_image)  # Resize sketch
    sketch_encoding = face_recognition.face_encodings(sketch_image)
    if not sketch_encoding:
        raise ValueError("No face found in the sketch!")
    sketch_encoding = sketch_encoding[0]  # Extract face encoding

    # Load images from the database
    database_folder = "database"
    database_images = [os.path.join(database_folder, name) for name in os.listdir(database_folder)]

    # Try to load cached encodings
    if os.path.exists("encodings.pkl"):
        logging.info("Loading cached encodings...")
        with open("encodings.pkl", "rb") as f:
            db_encodings = pickle.load(f)
    else:
        logging.info("Computing encodings for database images...")
        # Process images in parallel
        with Pool() as pool:
            results = pool.map(process_image, database_images)

        # Filter out None values (images with no faces)
        db_encodings = [result for result in results if result is not None]

        # Save encodings to a file
        with open("encodings.pkl", "wb") as f:
            pickle.dump(db_encodings, f)

    # Compare sketch with each image in the database
    best_match = None
    best_match_name = None
    best_match_distance = float("inf")
    MATCH_THRESHOLD = 0.6  # Adjust this threshold as needed

    logging.info(f"Processing {len(db_encodings)} images...")
    for i, (image_path, db_encoding) in enumerate(db_encodings):
        logging.info(f"Processing image {i + 1}/{len(db_encodings)}: {image_path}")
        distance = np.linalg.norm(sketch_encoding - db_encoding)
        logging.info(f"Distance: {distance}")

        # Update best match if distance is below the threshold
        if distance < MATCH_THRESHOLD and distance < best_match_distance:
            best_match_distance = distance
            best_match_name = os.path.basename(image_path)
            best_match = face_recognition.load_image_file(image_path)

    # Display the best match
    if best_match is not None:
        print(f"Best match: {best_match_name} (Distance: {best_match_distance})")
        cv2.imshow("Best Match", cv2.cvtColor(best_match, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No match found in the database!")

if __name__ == '__main__':
    freeze_support()  # Required for Windows
    main()