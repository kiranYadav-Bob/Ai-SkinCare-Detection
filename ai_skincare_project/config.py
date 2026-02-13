"""
Configuration file for AI Skincare Project
Contains all global settings, paths, and parameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv



# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
ROOT_DIR = PROJECT_ROOT

# Directory paths (updated for flat structure)
INPUT_IMAGE_DIR = PROJECT_ROOT / "uploads"
LOGS_DIR = PROJECT_ROOT / "logs"

# File paths
SKIN_ISSUES_CSV = PROJECT_ROOT / "skin_issues.csv"
LABEL_MAPPING_JSON = PROJECT_ROOT / "label_mapping.json"

# API Configuration (Optional - uses rule-based fallback if not provided)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4"
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.7

# Image Processing Configuration
IMAGE_SIZE = (224, 224)  # Standard size for CNN input
FACE_CROP_SIZE = (400, 400)  # Size for face cropping
SKIN_SEGMENTATION_THRESHOLD = 0.5
MAX_IMAGE_SIZE_MB = 10

# Model Configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

MODEL_WEIGHTS_DIR = PROJECT_ROOT / "model_weights"
MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)

MODEL_FILES = {
    "cnn_model": MODEL_WEIGHTS_DIR / "skin_cnn_model.h5",
    "skin_issue_model": MODEL_WEIGHTS_DIR / "skin_model.h5",  # update with actual filename
    "label_mapping": LABEL_MAPPING_JSON
}

# Skin Issues Labels
SKIN_ISSUES = [
    "acne",
    "blackheads",
    "whiteheads",
    "pimples",
    "dark_spots",
    "hyperpigmentation",
    "dry_skin",
    "oily_skin",
    "combination_skin",
    "sensitive_skin",
    "wrinkles",
    "fine_lines",
    "large_pores",
    "uneven_skin_tone",
    "redness",
    "inflammation",
    "scarring",
    "age_spots",
    "sun_damage",
    "normal_skin"
]


# Confidence thresholds
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.3  # Lowered for demo mode
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # Add this line - adjust value as needed


# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"


# API Endpoints (Optional - uses rule-based fallback if not available)
API_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions"
}


# UI Configuration
STREAMLIT_PAGE_TITLE = "AI Skincare Analysis"
STREAMLIT_PAGE_ICON = "🤖"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_SIDEBAR_STATE = "collapsed"

# File upload configuration
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB



# Create directories if they don't exist
def create_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        INPUT_IMAGE_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories()





# Error messages
ERROR_MESSAGES = {
    "no_face_detected": "No face detected in the image. Please upload a clear face photo.",
    "image_too_large": f"Image size exceeds {MAX_IMAGE_SIZE_MB}MB limit.",
    "invalid_image": "Invalid image format. Please upload JPEG or PNG.",
    "api_error": "Error connecting to AI service. Please try again.",
    "model_not_found": "AI model not found. Please check installation."
}

# Success messages
SUCCESS_MESSAGES = {
    "analysis_complete": "Skin analysis completed successfully!",
    "recommendations_generated": "Personalized recommendations generated!",
    "image_uploaded": "Image uploaded successfully!"
} 