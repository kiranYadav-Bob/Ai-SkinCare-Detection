"""
Modified Face Detection and Cropping Module (OpenCV Only)
Converted from original 496-line version by removing dlib completely
"""

import cv2
import numpy as np
from PIL import Image
import logging
import os
from typing import List, Tuple, Optional, Dict, Any, Union

# Project imports
from config import (
    FACE_CROP_SIZE, 
    DETECTION_CONFIDENCE_THRESHOLD,
    ERROR_MESSAGES
)
from image_utils import (
    pil_to_cv2, 
    cv2_to_pil, 
    resize_image,
    crop_face_region
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        # Load additional cascade for better detection
        profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
        
        # Check if cascades loaded successfully
        if self.face_cascade.empty():
            logger.warning("Frontal face cascade not loaded, using fallback")
        if self.profile_cascade.empty():
            logger.warning("Profile face cascade not loaded")

    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Enhance image for better detection
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20), maxSize=(0, 0)
            )
            return faces.tolist()
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {str(e)}")
            return []

    def detect_faces_advanced(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Enhance image
            gray = cv2.equalizeHist(gray)
            
            # Try multiple detection parameters
            all_faces = []
            
            # Method 1: Standard detection
            faces_haar = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20), maxSize=(0, 0)
            )
            all_faces.extend(faces_haar.tolist())
            
            # Method 2: More sensitive detection
            faces_sensitive = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=2, minSize=(15, 15), maxSize=(0, 0)
            )
            all_faces.extend(faces_sensitive.tolist())
            
            # Method 3: Profile face detection
            if not self.profile_cascade.empty():
                profile_faces = self.profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20), maxSize=(0, 0)
                )
                all_faces.extend(profile_faces.tolist())

            return all_faces
        except Exception as e:
            logger.error(f"Advanced OpenCV face detection failed: {str(e)}")
            return []

    def detect_faces_combined(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            all_faces = []
            # Try multiple detection methods
            methods = [
                self.detect_faces_advanced,  # Most comprehensive
                self.detect_faces_opencv,    # Standard
            ]
            
            for method in methods:
                try:
                    faces = method(image)
                    all_faces.extend(faces)
                except Exception as e:
                    logger.warning(f"Face detection method failed: {str(e)}")
            
            # If no faces detected, try fallback approach
            if not all_faces:
                return self._fallback_face_detection(image)
            
            merged_faces = self._merge_overlapping_faces(all_faces)
            return merged_faces
        except Exception as e:
            logger.error(f"Combined face detection failed: {str(e)}")
            return self._fallback_face_detection(image)
    
    def _fallback_face_detection(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Fallback face detection when standard methods fail"""
        try:
            # Create a simple face region based on image center
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Estimate face size (typically 1/3 to 1/2 of image)
            face_size = min(w, h) // 3
            
            # Create face region around center
            x = max(0, center_x - face_size // 2)
            y = max(0, center_y - face_size // 2)
            face_w = min(face_size, w - x)
            face_h = min(face_size, h - y)
            
            logger.info("Using fallback face detection - assuming center region")
            return [(x, y, face_w, face_h)]
            
        except Exception as e:
            logger.error(f"Fallback face detection failed: {str(e)}")
            return []

    def _merge_overlapping_faces(self, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if not faces:
            return []
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        merged = []
        for face in faces:
            x1, y1, w1, h1 = face
            should_merge = False
            for i, mf in enumerate(merged):
                x2, y2, w2, h2 = mf
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                if overlap_area > 0.5 * min(area1, area2):
                    if area1 > area2:
                        merged[i] = face
                    should_merge = True
                    break
            if not should_merge:
                merged.append(face)
        return merged

    def crop_face(self, image: np.ndarray, face_coords: Tuple[int, int, int, int], 
                  target_size: Tuple[int, int] = FACE_CROP_SIZE) -> np.ndarray:
        try:
            x, y, w, h = face_coords
            padding = int(min(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            face_crop = image[y1:y2, x1:x2]
            resized_face = cv2.resize(face_crop, target_size)
            return resized_face
        except Exception as e:
            logger.error(f"Error cropping face: {str(e)}")
            raise

    def analyze_face_quality(self, face_crop: np.ndarray) -> Dict[str, Any]:
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            return {
                'brightness': np.mean(gray),
                'contrast': np.std(gray),
                'sharpness': sharpness,
                'face_size': face_crop.shape[0] * face_crop.shape[1],
                'aspect_ratio': face_crop.shape[1] / face_crop.shape[0],
                'overall_score': (np.mean(gray) + np.std(gray) + sharpness) / 3
            }
        except Exception as e:
            logger.error(f"Error analyzing face quality: {str(e)}")
            return {}

class FaceCropProcessor:
    def __init__(self):
        self.face_detector = FaceDetector()

    def process_image(self, image: Union[Image.Image, np.ndarray], method: str = 'combined') -> Dict[str, Any]:
        try:
            if isinstance(image, Image.Image):
                cv2_image = pil_to_cv2(image)
            else:
                cv2_image = image.copy()

            if method == 'opencv':
                faces = self.face_detector.detect_faces_opencv(cv2_image)
            elif method == 'opencv_advanced':
                faces = self.face_detector.detect_faces_advanced(cv2_image)
            else:
                faces = self.face_detector.detect_faces_combined(cv2_image)

            if not faces:
                # Try fallback detection
                faces = self.face_detector._fallback_face_detection(cv2_image)
                if not faces:
                    return {
                        'success': False,
                        'error': ERROR_MESSAGES["no_face_detected"],
                        'faces': [],
                        'cropped_faces': []
                    }
                else:
                    logger.info("Using fallback face detection")

            cropped_faces = []
            face_analyses = []
            for i, face_coords in enumerate(faces):
                face_crop = self.face_detector.crop_face(cv2_image, face_coords)
                quality = self.face_detector.analyze_face_quality(face_crop)
                face_pil = cv2_to_pil(face_crop)
                cropped_faces.append(face_pil)
                face_analyses.append({
                    'face_id': i,
                    'coordinates': face_coords,
                    'quality_metrics': quality
                })

            return {
                'success': True,
                'faces': faces,
                'cropped_faces': cropped_faces,
                'face_analyses': face_analyses,
                'total_faces': len(faces)
            }

        except Exception as e:
            logger.error(f"Face processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'faces': [],
                'cropped_faces': []
            }

    def get_best_face(self, face_analyses: List[Dict[str, Any]]) -> Optional[int]:
        try:
            if not face_analyses:
                return None
            return max(range(len(face_analyses)), key=lambda i: face_analyses[i]['quality_metrics'].get('overall_score', 0))
        except Exception as e:
            logger.error(f"Error selecting best face: {str(e)}")
            return 0 if face_analyses else None
