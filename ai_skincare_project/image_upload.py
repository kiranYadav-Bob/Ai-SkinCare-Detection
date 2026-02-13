"""
Image Upload Module for AI Skincare Project
Handles image upload, validation, and preprocessing for skin analysis
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import logging

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALLOWED_IMAGE_TYPES, 
    MAX_FILE_SIZE, 
    ERROR_MESSAGES, 
    SUCCESS_MESSAGES,
    INPUT_IMAGE_DIR
)
from image_utils import (
    validate_image, 
    bytes_to_pil_image, 
    resize_image,
    image_to_base64,
    calculate_image_statistics
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageUploadHandler:
    """
    Handles image upload functionality for the skincare analysis app
    """
    
    def __init__(self):
        self.uploaded_files = []
        self.processed_images = []
        self.upload_metadata = {}
    
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """Public wrapper to process a single uploaded file.
        This delegates to the internal implementation to keep
        backward compatibility with callers expecting a public API.
        """
        return self._process_uploaded_file(uploaded_file)
    
    def upload_interface(self) -> Optional[Dict[str, Any]]:
        """
        Main upload interface using Streamlit
        
        Returns:
            Dictionary containing uploaded image data and metadata
        """
        st.header("📸 Upload Your Skin Image")
        st.markdown("Upload a clear photo of your face for skin analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear face photo (JPEG, PNG up to 10MB)"
        )
        
        if uploaded_file is not None:
            return self._process_uploaded_file(uploaded_file)
        
        return None
    
    def _process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process uploaded file and return processed data
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with processed image data
        """
        try:
            # Read file data
            file_data = uploaded_file.read()
            filename = uploaded_file.name
            
            # Validate image
            is_valid, error_message = validate_image(file_data, filename)
            
            if not is_valid:
                st.error(error_message)
                return None
            
            # Convert to PIL image
            pil_image = bytes_to_pil_image(file_data)
            
            # Display original image
            st.subheader("📷 Original Image")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(pil_image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                # Image information
                st.write("**Image Details:**")
                st.write(f"📁 Filename: {filename}")
                st.write(f"📏 Size: {pil_image.size[0]} x {pil_image.size[1]}")
                st.write(f"💾 File size: {len(file_data) / 1024 / 1024:.2f} MB")
                st.write(f"🎨 Mode: {pil_image.mode}")
                
                # Image statistics
                from image_utils import pil_to_cv2
                image_array = pil_to_cv2(pil_image)
                stats = calculate_image_statistics(image_array)
                st.write("**Image Statistics:**")
                st.write(f"📊 Mean: {stats['mean']:.2f}")
                st.write(f"📈 Std: {stats['std']:.2f}")
            
            # Image enhancement options
            st.subheader("🔧 Image Enhancement")
            enhance_image = st.checkbox("Enhance image quality", value=True)
            
            if enhance_image:
                pil_image = self._enhance_image_quality(pil_image)
                st.success("✅ Image enhanced successfully!")
            
            # Save processed image
            processed_data = self._save_processed_image(pil_image, filename)
            
            # Add required fields for app compatibility
            processed_data['success'] = True
            processed_data['pil_image'] = pil_image
            
            # Success message
            st.success(SUCCESS_MESSAGES["image_uploaded"])
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            st.error(f"Error processing image: {str(e)}")
            return None
    
    def _enhance_image_quality(self, image) -> Any:
        """
        Enhance image quality for better analysis
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        try:
            from image_utils import enhance_image
            
            # Enhancement parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            
            with col2:
                contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            
            with col3:
                sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
            
            # Apply enhancements
            enhanced_image = enhance_image(
                image, 
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness
            )
            
            # Show before/after comparison
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Before Enhancement", use_container_width=True)
            with col2:
                st.image(enhanced_image, caption="After Enhancement", use_container_width=True)
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image
    
    def _save_processed_image(self, image, original_filename: str) -> Dict[str, Any]:
        """
        Save processed image and return metadata
        
        Args:
            image: PIL Image object
            original_filename: Original filename
            
        Returns:
            Dictionary with image data and metadata
        """
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate new filename
            name, ext = os.path.splitext(original_filename)
            new_filename = f"{name}_{timestamp}{ext}"
            
            # Ensure upload directory exists
            INPUT_IMAGE_DIR.mkdir(exist_ok=True)
            
            # Save image
            save_path = INPUT_IMAGE_DIR / new_filename
            image.save(save_path)
            
            # Convert to base64 for display
            base64_image = image_to_base64(image)
            
            # Create metadata
            metadata = {
                'original_filename': original_filename,
                'processed_filename': new_filename,
                'save_path': str(save_path),
                'timestamp': timestamp,
                'image_size': image.size,
                'file_size': os.path.getsize(save_path),
                'image_format': image.format if image.format else ext.replace(".", "").upper(),
                'base64_image': base64_image
            }
            
            # Store in session state
            if 'uploaded_images' not in st.session_state:
                st.session_state.uploaded_images = []
            
            st.session_state.uploaded_images.append(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error saving processed image: {str(e)}")
            raise
    
    def batch_upload_interface(self) -> list:
        """
        Interface for batch image upload
        
        Returns:
            List of processed image data
        """
        st.header("📁 Batch Image Upload")
        st.markdown("Upload multiple images for batch analysis")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple face photos (JPEG, PNG up to 10MB each)"
        )
        
        processed_images = []
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    processed_data = self._process_uploaded_file(uploaded_file)
                    if processed_data:
                        processed_images.append(processed_data)
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            status_text.text("✅ Batch processing completed!")
            st.success(f"Successfully processed {len(processed_images)} images")
        
        return processed_images
    
    def camera_upload_interface(self) -> Optional[Dict[str, Any]]:
        """
        Interface for camera capture
        
        Returns:
            Dictionary with captured image data
        """
        st.header("📱 Camera Capture")
        st.markdown("Take a photo using your camera")
        
        camera_photo = st.camera_input("Take a photo")
        
        if camera_photo is not None:
            # Convert to bytes
            file_data = camera_photo.read()
            
            # Process as regular upload
            from io import BytesIO
            uploaded_file = type('UploadedFile', (), {
                'read': lambda: file_data,
                'name': f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            })()
            
            return self._process_uploaded_file(uploaded_file)
        
        return None
    
    def get_upload_history(self) -> list:
        """
        Get upload history from session state
        
        Returns:
            List of uploaded images
        """
        return st.session_state.get('uploaded_images', [])
    
    def clear_upload_history(self):
        """Clear upload history"""
        if 'uploaded_images' in st.session_state:
            del st.session_state.uploaded_images
        st.success("Upload history cleared!")

def main():
    """
    Main function for testing the upload module
    """
    st.set_page_config(
        page_title="Image Upload - AI Skincare",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("🧴 AI Skincare - Image Upload")
    
    # Initialize upload handler
    upload_handler = ImageUploadHandler()
    
    # Upload method selection
    upload_method = st.sidebar.selectbox(
        "Choose upload method:",
        ["Single Upload", "Batch Upload", "Camera Capture"]
    )
    
    if upload_method == "Single Upload":
        result = upload_handler.upload_interface()
        if result:
            st.json(result)
    
    elif upload_method == "Batch Upload":
        results = upload_handler.batch_upload_interface()
        if results:
            st.write(f"Processed {len(results)} images")
    
    elif upload_method == "Camera Capture":
        result = upload_handler.camera_upload_interface()
        if result:
            st.json(result)
    
    # Upload history
    st.sidebar.subheader("📚 Upload History")
    history = upload_handler.get_upload_history()
    
    if history:
        for i, item in enumerate(history):
            st.sidebar.write(f"{i+1}. {item['original_filename']}")
        
        if st.sidebar.button("Clear History"):
            upload_handler.clear_upload_history()
    else:
        st.sidebar.write("No upload history")

if __name__ == "__main__":
    main() 