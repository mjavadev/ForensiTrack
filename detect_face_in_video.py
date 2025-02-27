import cv2
import numpy as np
import torch
import timm
import mediapipe as mp
import open3d as o3d
from deepface import DeepFace
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
from torchvision import transforms
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Dict
from face3d.morphable_model import MorphabelModel

class ReIDModel(torch.nn.Module):
    def __init__(self, model_name: str = "osnet_x1_0"):
        super(ReIDModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(image_rgb).unsqueeze(0)
            with torch.no_grad():
                features = self.model(img_tensor)
            return features.cpu().numpy()
        except Exception as e:
            print(f"ReID extraction error: {e}")
            return None

class GaitAnalyzer:
    def __init__(self, buffer_size: int = 30):
        self.buffer = deque(maxlen=buffer_size)
        self.pose_landmarks = mp.solutions.pose.PoseLandmark
    
    def add_pose(self, landmarks):
        self.buffer.append(landmarks)
    
    def analyze_gait(self) -> Optional[dict]:
        if len(self.buffer) < self.buffer.maxlen:
            return None
        left_ankle = [pose[self.pose_landmarks.LEFT_ANKLE.value] for pose in self.buffer]
        right_ankle = [pose[self.pose_landmarks.RIGHT_ANKLE.value] for pose in self.buffer]
        stride_length = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))
        return {"stride_length": stride_length}

class EnhancedDetector:
    def __init__(self):
        self.face_detector = DeepFace.build_model("ArcFace")
        self.reid_model = ReIDModel()
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.yolo_model = YOLO("yolov8n.pt")
        self.gait_analyzer = GaitAnalyzer()
        self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3, nn_budget=100)
        self.bfm = MorphabelModel('BFM_model.mat')
    
    def extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            face_objs = DeepFace.extract_faces(img_path=image, detector_backend="retinaface", enforce_detection=False)
            if not face_objs:
                return None
            face_img = cv2.resize(face_objs[0]["face"], (112, 112))
            face_img = (face_img.astype(np.float32) / 255.0 - 0.5) * 2.0
            return self.face_detector.predict(face_img.reshape(1, 112, 112, 3))[0]
        except Exception as e:
            print(f"Face feature extraction error: {e}")
            return None

    def reconstruct_3d_face_3DMM(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            shape_params, exp_params = self.bfm.fit(gray)
            return self.bfm.generate_vertices(shape_params, exp_params)
        except Exception as e:
            print(f"3D Face Reconstruction (3DMM) error: {e}")
            return None
    
    def detect_and_track(self, frame: np.ndarray):
        try:
            results = self.yolo_model(frame)
            detections = []
            for result in results:
                for box in result.boxes.xyxy.cpu().numpy():
                    detections.append(box)
            return self.tracker.update(detections, frame)
        except Exception as e:
            print(f"Detection and tracking error: {e}")
            return {}

def compare_3d_faces(face1: np.ndarray, face2: np.ndarray) -> float:
    try:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(face1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(face2)
        distance = np.asarray(pcd1.compute_point_cloud_distance(pcd2)).mean()
        return 1 - (distance / np.max(distance))
    except Exception as e:
        print(f"Point Cloud Comparison Error: {e}")
        return 0.0

def process_frame(detector: EnhancedDetector, frame: np.ndarray, frame_index: int):
    detected_ids = detector.detect_and_track(frame)
    suspects = {}
    for track_id, box in detected_ids.items():
        x1, y1, x2, y2 = map(int, box)
        cropped_person = frame[y1:y2, x1:x2]
        face_features = detector.extract_face_features(cropped_person)
        if face_features is None:
            face_features = detector.reconstruct_3d_face_3DMM(cropped_person)
        reid_features = detector.reid_model.extract_features(cropped_person)
        pose_results = detector.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if pose_results.pose_landmarks:
            detector.gait_analyzer.add_pose(pose_results.pose_landmarks)
        suspects[track_id] = {"face": face_features, "reid": reid_features, "gait": detector.gait_analyzer.analyze_gait()}
    return suspects

def detect_suspect(video_path: str, suspect_image_path: str):
    detector = EnhancedDetector()
    ref_face_features = detector.extract_face_features(cv2.imread(suspect_image_path))
    ref_3d_face = detector.reconstruct_3d_face_3DMM(cv2.imread(suspect_image_path))
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        suspects = process_frame(detector, frame, 0)
        for track_id, features in suspects.items():
            if features["face"] is not None and ref_3d_face is not None:
                score = compare_3d_faces(features["face"], ref_3d_face)
                if score > 0.6:
                    print(f"Suspect detected! ID: {track_id}")
                    cap.release()
                    return True
    cap.release()
    return False
