from deepface import DeepFace
import cv2
import os
import numpy as np
import pickle
from scipy.spatial.distance import cosine
from multiprocessing import Pool, freeze_support
import logging

# Configure logging to print messages to the console
# Configure logging to print messages to the console
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.DEBUG)  # Set the root logger to DEBUG level

# Create a stream handler to output logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the level to DEBUG for console output

# Define log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the root logger
logger.addHandler(console_handler)


# Preprocessing function for sketches
def preprocess_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Enhance contrast
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)  # Smooth the image
    edges = cv2.Canny(blurred, 100, 200)  # Detect edges
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert back to RGB

# Normalize embeddings for consistency
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

# Process a single image and extract face embeddings
def process_image(image_path):
    try:
        embedding_facenet = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        embedding_arcface = DeepFace.represent(img_path=image_path, model_name="ArcFace", enforce_detection=False)

        facenet_embedding = normalize_embedding(np.array(embedding_facenet[0]["embedding"]))
        arcface_embedding = normalize_embedding(np.array(embedding_arcface[0]["embedding"]))

        return image_path, {"Facenet": facenet_embedding, "ArcFace": arcface_embedding}
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None

def main():
    # Load the sketch
    sketch_path = "vk.jpg"
    logging.info(f"Processing sketch: {sketch_path}")
    sketch_image = cv2.imread(sketch_path)
    sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2RGB)
    sketch_image = preprocess_sketch(sketch_image)

    # Extract embeddings for the sketch
    sketch_embedding_facenet = DeepFace.represent(img_path=sketch_image, model_name="Facenet", enforce_detection=False)
    sketch_embedding_arcface = DeepFace.represent(img_path=sketch_image, model_name="ArcFace", enforce_detection=False)

    if not sketch_embedding_facenet or not sketch_embedding_arcface:
        raise ValueError("No face found in the sketch!")

    # Normalize sketch embeddings
    sketch_embedding_facenet = normalize_embedding(np.array(sketch_embedding_facenet[0]["embedding"]))
    sketch_embedding_arcface = normalize_embedding(np.array(sketch_embedding_arcface[0]["embedding"]))

    # Load images from the database
    database_folder = "database"
    database_images = [os.path.join(database_folder, name) for name in os.listdir(database_folder)]

    # Try to load cached embeddings
    if os.path.exists("embeddings.pkl"):
        logging.info("Loading cached embeddings...")
        with open("embeddings.pkl", "rb") as f:
            db_embeddings = pickle.load(f)
    else:
        logging.info("Computing embeddings for database images...")
        with Pool() as pool:
            results = pool.map(process_image, database_images)

        db_embeddings = [result for result in results if result is not None]

        with open("embeddings.pkl", "wb") as f:
            pickle.dump(db_embeddings, f)

    # Match Parameters
    MATCH_THRESHOLD = 0.2  # Stricter match threshold
    REJECT_THRESHOLD = 0.15  # Stricter rejection

    best_match = None
    best_match_name = None
    best_match_distance = float("inf")

    logging.info(f"Processing {len(db_embeddings)} images...")
    for i, (image_path, db_embedding) in enumerate(db_embeddings):
        logging.info(f"Processing image {i + 1}/{len(db_embeddings)}: {image_path}")

        # Compute cosine distances
        distance_facenet = cosine(sketch_embedding_facenet, db_embedding["Facenet"])
        distance_arcface = cosine(sketch_embedding_arcface, db_embedding["ArcFace"])
        avg_distance = (distance_facenet + distance_arcface) / 2  

        # Convert to match confidence (higher is better)
        match_confidence = 1 - avg_distance  # 1 - distance to get confidence

        # Logging the embedding and distances for debugging
        logging.debug(f"Embeddings for sketch: {sketch_embedding_facenet[:5]}, {sketch_embedding_arcface[:5]}")
        logging.debug(f"Embeddings for database image {image_path}: {db_embedding['Facenet'][:5]}, {db_embedding['ArcFace'][:5]}")
        logging.debug(f"Cosine distances - Facenet: {distance_facenet}, ArcFace: {distance_arcface}, Avg Distance: {avg_distance}, Match Confidence: {match_confidence}")

        # Update best match if better
        if avg_distance < MATCH_THRESHOLD and avg_distance < best_match_distance:
            best_match_distance = avg_distance
            best_match_name = os.path.basename(image_path)
            best_match = cv2.imread(image_path)

    # Display results
    if best_match is not None:
        if best_match_distance < REJECT_THRESHOLD:
            print(f"Best match: {best_match_name} (Match Confidence: {1 - best_match_distance})")
            cv2.imshow("Best Match", best_match)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No match found in the database (rejected unknown face).")
    else:
        print("No match found in the database!")

if __name__ == '__main__':
    freeze_support()
    main()
