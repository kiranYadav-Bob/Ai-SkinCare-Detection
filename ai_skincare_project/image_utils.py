"""
Image utilities for AI Skincare Project
Contains functions for image processing, validation, and manipulation
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
from typing import Tuple, Optional, Union
import logging
from pathlib import Path

from config import (
    IMAGE_SIZE, 
    FACE_CROP_SIZE, 
    ALLOWED_IMAGE_TYPES, 
    MAX_FILE_SIZE,
    ERROR_MESSAGES
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_image(image_data: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate uploaded image file
    
    Args:
        image_data: Raw image data
        filename: Name of the uploaded file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file size
        if len(image_data) > MAX_FILE_SIZE:
            return False, ERROR_MESSAGES["image_too_large"]
        
        # Check file extension
        file_extension = Path(filename).suffix.lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            return False, ERROR_MESSAGES["invalid_image"]
        
        # Try to open image to validate format
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False, ERROR_MESSAGES["invalid_image"]

def bytes_to_pil_image(image_data: bytes) -> Image.Image:
    """
    Convert bytes to PIL Image
    
    Args:
        image_data: Raw image data
        
    Returns:
        PIL Image object
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Error converting bytes to PIL image: {str(e)}")
        raise

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image array (BGR format)
    """
    try:
        # Convert PIL to numpy array
        numpy_image = np.array(pil_image)
        # Convert RGB to BGR for OpenCV
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return cv2_image
    except Exception as e:
        logger.error(f"Error converting PIL to CV2: {str(e)}")
        raise

def normalize_image(image: Union[Image.Image, np.ndarray], target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Normalize image for model input.

    Args:
        image: Input image (PIL or OpenCV format)
        target_size: Size to resize (width, height)

    Returns:
        Normalized NumPy array with pixel values in [0,1]
    """
    try:
        # Convert to PIL if OpenCV format
        if isinstance(image, np.ndarray):
            image = cv2_to_pil(image)

        # Resize image to target size without aspect ratio
        image = resize_image(image, target_size, maintain_aspect_ratio=False)

        # Convert to NumPy and normalize to [0, 1]
        img_array = np.array(image).astype('float32') / 255.0

        return img_array
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        raise


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV (BGR) image to PIL (RGB) format.
    
    Args:
        cv2_image: OpenCV image array (BGR format)
    
    Returns:
        PIL Image object (RGB format)
    """
    try:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    except Exception as e:
        logger.error(f"Error converting CV2 to PIL: {str(e)}")
        raise


def crop_face_region(cv2_image: np.ndarray, face_bbox: tuple) -> np.ndarray:
    """
    Crop a face region from an OpenCV image.
    
    Args:
        cv2_image: OpenCV image array (BGR format)
        face_bbox: Tuple of (x, y, w, h) for the face region
    
    Returns:
        Cropped face image as OpenCV array (BGR format)
    """
    try:
        x, y, w, h = face_bbox
        cropped = cv2_image[y:y+h, x:x+w]
        return cropped
    except Exception as e:
        logger.error(f"Error cropping face region: {str(e)}")
        raise




def resize_image(image: Union[Image.Image, np.ndarray], 
                target_size: Tuple[int, int] = IMAGE_SIZE,
                maintain_aspect_ratio: bool = True) -> Union[Image.Image, np.ndarray]:
    """
    Resize image to target size
    
    Args:
        image: Input image (PIL or CV2)
        target_size: Target dimensions (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    try:
        if isinstance(image, Image.Image):
            if maintain_aspect_ratio:
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            return image
        elif isinstance(image, np.ndarray):
            if maintain_aspect_ratio:
                h, w = image.shape[:2]
                aspect = w / h
                target_w, target_h = target_size
                target_aspect = target_w / target_h
                
                if aspect > target_aspect:
                    new_w = target_w
                    new_h = int(target_w / aspect)
                else:
                    new_h = target_h
                    new_w = int(target_h * aspect)
                
                resized = cv2.resize(image, (new_w, new_h))
            else:
                resized = cv2.resize(image, target_size)
            return resized
        else:
            raise ValueError("Image must be PIL Image or numpy array")
            
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise



def enhance_image(image: Image.Image, 
                 brightness: float = 1.0,
                 contrast: float = 1.0,
                 sharpness: float = 1.0) -> Image.Image:
    """
    Enhance image quality
    
    Args:
        image: Input PIL image
        brightness: Brightness factor (0.0 to 2.0)
        contrast: Contrast factor (0.0 to 2.0)
        sharpness: Sharpness factor (0.0 to 2.0)
        
    Returns:
        Enhanced image
    """
    try:
        enhanced = image.copy()
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        raise

def image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """
    Convert PIL image to base64 string
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG)
        
    Returns:
        Base64 encoded string
    """
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise







def calculate_image_statistics(image: np.ndarray) -> dict:
    """
    Calculate basic image statistics
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary with image statistics
    """
    try:
        stats = {
            'mean': np.mean(image),
            'std': np.std(image),
            'min': np.min(image),
            'max': np.max(image),
            'shape': image.shape
        }
        
        if len(image.shape) == 3:
            # Color image - calculate per channel
            stats['mean_rgb'] = np.mean(image, axis=(0, 1)).tolist()
            stats['std_rgb'] = np.std(image, axis=(0, 1)).tolist()
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating image statistics: {str(e)}")
        raise

def save_image_with_metadata(image: Union[Image.Image, np.ndarray], 
                           filepath: str, 
                           metadata: dict = None) -> bool:
    """
    Save image with optional metadata
    
    Args:
        image: Image to save
        filepath: Output file path
        metadata: Optional metadata dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if isinstance(image, np.ndarray):
            image = cv2_to_pil(image)
        
        # Save image
        image.save(filepath)
        
        # Save metadata if provided
        if metadata:
            metadata_path = str(filepath) + '.json'
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return False

def load_image_with_metadata(filepath: str) -> Tuple[Union[Image.Image, np.ndarray], dict]:
    """
    Load image with metadata
    
    Args:
        filepath: Input file path
        
    Returns:
        Tuple of (image, metadata)
    """
    try:
        # Load image
        image = Image.open(filepath)
        
        # Load metadata if exists
        metadata = {}
        metadata_path = str(filepath) + '.json'
        if Path(metadata_path).exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return image, metadata
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise 