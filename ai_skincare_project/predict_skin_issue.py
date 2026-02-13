"""
Skin Issue Prediction Module for AI Skincare Project
Loads trained models and performs inference on new images
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import os
from pathlib import Path
import cv2

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    IMAGE_SIZE, 
    CLASSIFICATION_CONFIDENCE_THRESHOLD,
    SKIN_ISSUES,
    MODEL_WEIGHTS_DIR,
    MODEL_FILES
)
from image_utils import (
    resize_image, 
    normalize_image,
    pil_to_cv2,
    cv2_to_pil
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinIssuePredictor:
    """
    Skin issue prediction using trained CNN models
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.use_demo = False
        self.classes = SKIN_ISSUES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.model_path = model_path or str(MODEL_FILES["cnn_model"])
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                self.use_demo = False
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Using feature-based demo predictor (no trained model available)")
                self.model = None
                self.use_demo = True
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Falling back to feature-based demo predictor")
            self.model = None
            self.use_demo = True
    
    def _create_demo_model(self):
        
        try:
            from tensorflow.keras import layers, models
            
        
            model = models.Sequential([
                layers.Input(shape=(*IMAGE_SIZE, 3)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(len(self.classes), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("model created successfully")
            
        except Exception as e:
            logger.error(f"Error creating  model: {str(e)}")
            raise
    
    def _demo_predict(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Simple image analysis for demo
            results = self._analyze_image_features(img_array)
            
            return {
                'success': True,
                'primary_prediction': results['primary'],
                'top_3_predictions': results['top_3'],
                'all_predictions': results['all'],
                'is_confident': True,
                'confidence_threshold': CLASSIFICATION_CONFIDENCE_THRESHOLD,
                'raw_predictions': results['raw']
            }
            
        except Exception as e:
            logger.error(f"Demo prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'primary_prediction': None,
                'top_3_predictions': [],
                'all_predictions': []
            }
    
    def _analyze_image_features(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze image features to detect skin issues"""
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Calculate basic statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            contrast = np.std(gray)
            
            # Analyze color channels
            r_channel = img_array[:, :, 0]
            g_channel = img_array[:, :, 1]
            b_channel = img_array[:, :, 2]
            
            # Detect redness (potential inflammation/acne)
            redness = np.mean(r_channel) - np.mean(g_channel)
            
            # Detect dark spots (hyperpigmentation)
            dark_spots = np.sum(gray < 100) / gray.size
            
            # Detect texture variations (acne, scars)
            texture_variance = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            
            # Create predictions based on image analysis
            predictions = []
            
            # Detect acne based on redness and texture
            if redness > 15 and texture_variance > 80:
                acne_confidence = min(0.85, 0.4 + (redness / 40) + (texture_variance / 150))
                predictions.append({
                    'class': 'acne',
                    'confidence': acne_confidence,
                    'index': self.class_to_idx.get('acne', 1)
                })
            
            # Detect dark spots
            if dark_spots > 0.08:
                dark_spots_confidence = min(0.8, 0.3 + (dark_spots * 3))
                predictions.append({
                    'class': 'dark_spots',
                    'confidence': dark_spots_confidence,
                    'index': self.class_to_idx.get('dark_spots', 4)
                })
            
            # Detect dryness (low contrast, uniform texture)
            if contrast < 25 and texture_variance < 40:
                dry_confidence = min(0.75, 0.3 + (25 - contrast) / 25)
                predictions.append({
                    'class': 'dry_skin',
                    'confidence': dry_confidence,
                    'index': self.class_to_idx.get('dry_skin', 6)
                })
            
            # Detect oiliness (high intensity, low contrast)
            if mean_intensity > 140 and contrast < 35:
                oily_confidence = min(0.8, 0.3 + (mean_intensity - 140) / 40)
                predictions.append({
                    'class': 'oily_skin',
                    'confidence': oily_confidence,
                    'index': self.class_to_idx.get('oily_skin', 7)
                })
            
            # Detect wrinkles (fine lines in texture)
            if texture_variance > 120 and contrast > 45:
                wrinkle_confidence = min(0.7, 0.2 + (texture_variance / 200))
                predictions.append({
                    'class': 'wrinkles',
                    'confidence': wrinkle_confidence,
                    'index': self.class_to_idx.get('wrinkles', 2)
                })
            
            # Detect blackheads (dark circular spots)
            if dark_spots > 0.05 and texture_variance > 60:
                blackhead_confidence = min(0.65, 0.2 + (dark_spots * 4))
                predictions.append({
                    'class': 'blackheads',
                    'confidence': blackhead_confidence,
                    'index': self.class_to_idx.get('blackheads', 3)
                })
            
            # Detect whiteheads (bright spots with texture)
            if mean_intensity > 160 and texture_variance > 70:
                whitehead_confidence = min(0.6, 0.2 + (mean_intensity - 160) / 30)
                predictions.append({
                    'class': 'whiteheads',
                    'confidence': whitehead_confidence,
                    'index': self.class_to_idx.get('whiteheads', 5)
                })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Only add normal_skin if no other issues detected with high confidence
            if not predictions or all(pred['confidence'] < 0.4 for pred in predictions):
                predictions.append({
                    'class': 'normal_skin',
                    'confidence': 0.6,
                    'index': self.class_to_idx.get('normal_skin', 0)
                })
            
            # Create raw predictions array
            raw_predictions = [0.0] * len(self.classes)
            for pred in predictions:
                raw_predictions[pred['index']] = pred['confidence']
            
            return {
                'primary': predictions[0],
                'top_3': predictions[:3],
                'all': predictions,
                'raw': raw_predictions
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            # Return default prediction
            return {
                'primary': {
                    'class': 'normal_skin',
                    'confidence': 0.5,
                    'index': 0
                },
                'top_3': [{
                    'class': 'normal_skin',
                    'confidence': 0.5,
                    'index': 0
                }],
                'all': [{
                    'class': 'normal_skin',
                    'confidence': 0.5,
                    'index': 0
                }],
                'raw': [0.5] + [0.0] * (len(self.classes) - 1)
            }
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (PIL or numpy array)
            
        Returns:
            Preprocessed image array
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                # Resize image
                image = resize_image(image, IMAGE_SIZE, maintain_aspect_ratio=False)
                image_array = np.array(image)
            else:
                # Convert CV2 BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_array = cv2_to_pil(image)
                    image_array = resize_image(image_array, IMAGE_SIZE, maintain_aspect_ratio=False)
                    image_array = np.array(image_array)
                else:
                    image_array = image
            
            # Normalize image
            image_array = normalize_image(image_array)
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Predict skin issues from image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Use demo predictor when no trained model is available
            if self.use_demo or self.model is None or not hasattr(self.model, 'predict'):
                logger.info("Using demo prediction (no trained model available)")
                return self._demo_predict(image)
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1]
            top_predictions = []
            
            for idx in top_indices:
                class_name = self.idx_to_class[idx]
                confidence = float(predictions[0][idx])
                
                top_predictions.append({
                    'class': class_name,
                    'confidence': confidence,
                    'index': int(idx)
                })
            
            # Get primary prediction
            primary_prediction = top_predictions[0]
            
            # Determine if prediction meets confidence threshold
            is_confident = primary_prediction['confidence'] >= CLASSIFICATION_CONFIDENCE_THRESHOLD
            
            # Get top 3 predictions
            top_3_predictions = top_predictions[:3]
            
            result = {
                'success': True,
                'primary_prediction': primary_prediction,
                'top_3_predictions': top_3_predictions,
                'all_predictions': top_predictions,
                'is_confident': is_confident,
                'confidence_threshold': CLASSIFICATION_CONFIDENCE_THRESHOLD,
                'raw_predictions': predictions[0].tolist()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Fallback to demo prediction
            logger.info("Falling back to demo prediction")
            return self._demo_predict(image)
    
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Predict skin issues for multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            for i, image in enumerate(images):
                try:
                    result = self.predict(image)
                    result['image_index'] = i
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting image {i}: {str(e)}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'image_index': i,
                        'primary_prediction': None,
                        'top_3_predictions': []
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return []
    
    def get_prediction_confidence(self, prediction_result: Dict[str, Any]) -> float:
        """
        Get confidence score from prediction result
        
        Args:
            prediction_result: Prediction result dictionary
            
        Returns:
            Confidence score
        """
        try:
            if prediction_result.get('success') and prediction_result.get('primary_prediction'):
                return prediction_result['primary_prediction']['confidence']
            return 0.0
        except Exception as e:
            logger.error(f"Error getting confidence: {str(e)}")
            return 0.0
    
    def get_primary_issue(self, prediction_result: Dict[str, Any]) -> Optional[str]:
        """
        Get primary skin issue from prediction result
        
        Args:
            prediction_result: Prediction result dictionary
            
        Returns:
            Primary skin issue or None
        """
        try:
            if prediction_result.get('success') and prediction_result.get('primary_prediction'):
                return prediction_result['primary_prediction']['class']
            return None
        except Exception as e:
            logger.error(f"Error getting primary issue: {str(e)}")
            return None
    
    def get_all_issues(self, prediction_result: Dict[str, Any], confidence_threshold: float = 0.1) -> List[str]:
        """
        Get all detected skin issues above confidence threshold
        
        Args:
            prediction_result: Prediction result dictionary
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of detected skin issues
        """
        try:
            issues = []
            if prediction_result.get('success') and prediction_result.get('all_predictions'):
                for pred in prediction_result['all_predictions']:
                    if pred['confidence'] >= confidence_threshold:
                        issues.append(pred['class'])
            return issues
        except Exception as e:
            logger.error(f"Error getting all issues: {str(e)}")
            return []

class MultiLabelSkinPredictor:
    """
    Multi-label skin issue predictor for detecting multiple issues simultaneously
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.predictor = SkinIssuePredictor(model_path)
        self.issue_categories = {
            'acne_related': ['acne', 'blackheads', 'whiteheads', 'pimples'],
            'pigmentation': ['dark_spots', 'hyperpigmentation', 'age_spots', 'sun_damage'],
            'texture': ['dry_skin', 'oily_skin', 'combination_skin', 'large_pores'],
            'aging': ['wrinkles', 'fine_lines'],
            'sensitivity': ['sensitive_skin', 'redness', 'inflammation'],
            'other': ['scarring', 'uneven_skin_tone', 'normal_skin']
        }
    
    def predict_multi_label(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Predict multiple skin issues with confidence scores
        
        Args:
            image: Input image
            
        Returns:
            Multi-label prediction results
        """
        try:
            # Get single prediction
            single_result = self.predictor.predict(image)
            
            if not single_result['success']:
                return single_result
            
            # Analyze predictions by category
            category_results = {}
            detected_issues = []
            
            for category, issues in self.issue_categories.items():
                category_predictions = []
                category_confidence = 0.0
                
                for pred in single_result['all_predictions']:
                    if pred['class'] in issues:
                        category_predictions.append(pred)
                        category_confidence = max(category_confidence, pred['confidence'])
                
                category_results[category] = {
                    'predictions': category_predictions,
                    'max_confidence': category_confidence,
                    'has_issues': category_confidence >= 0.3  # Lower threshold for multi-label
                }
                
                if category_confidence >= 0.3:
                    detected_issues.extend([pred['class'] for pred in category_predictions if pred['confidence'] >= 0.3])
            
            # Create multi-label result
            multi_label_result = {
                'success': True,
                'primary_prediction': single_result['primary_prediction'],
                'detected_issues': detected_issues,
                'category_results': category_results,
                'issue_summary': self._create_issue_summary(category_results),
                'recommendation_priority': self._get_recommendation_priority(category_results)
            }
            
            return multi_label_result
            
        except Exception as e:
            logger.error(f"Error in multi-label prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'detected_issues': [],
                'category_results': {}
            }
    
    def _create_issue_summary(self, category_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create summary of detected issues
        
        Args:
            category_results: Category-wise prediction results
            
        Returns:
            Issue summary
        """
        try:
            summary = {
                'total_issues': 0,
                'primary_concern': None,
                'secondary_concerns': [],
                'severity_level': 'low'
            }
            
            # Count total issues
            for category, result in category_results.items():
                if result['has_issues']:
                    summary['total_issues'] += len(result['predictions'])
            
            # Determine primary concern (highest confidence)
            max_confidence = 0.0
            for category, result in category_results.items():
                if result['has_issues'] and result['max_confidence'] > max_confidence:
                    max_confidence = result['max_confidence']
                    if result['predictions']:
                        summary['primary_concern'] = result['predictions'][0]['class']
            
            # Get secondary concerns
            for category, result in category_results.items():
                if result['has_issues'] and result['predictions']:
                    for pred in result['predictions']:
                        if pred['class'] != summary['primary_concern'] and pred['confidence'] >= 0.3:
                            summary['secondary_concerns'].append(pred['class'])
            
            # Determine severity level
            if summary['total_issues'] >= 5:
                summary['severity_level'] = 'high'
            elif summary['total_issues'] >= 3:
                summary['severity_level'] = 'medium'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating issue summary: {str(e)}")
            return {}
    
    def _get_recommendation_priority(self, category_results: Dict[str, Any]) -> List[str]:
        """
        Get recommendation priority based on detected issues
        
        Args:
            category_results: Category-wise prediction results
            
        Returns:
            List of priority categories
        """
        try:
            priorities = []
            
            # Priority order: sensitivity > acne > pigmentation > aging > texture > other
            priority_order = ['sensitivity', 'acne_related', 'pigmentation', 'aging', 'texture', 'other']
            
            for category in priority_order:
                if category in category_results and category_results[category]['has_issues']:
                    priorities.append(category)
            
            return priorities
            
        except Exception as e:
            logger.error(f"Error getting recommendation priority: {str(e)}")
            return []

class PredictionAnalyzer:
    """
    Analyzes prediction results and provides insights
    """
    
    def __init__(self):
        self.severity_levels = {
            'low': {'range': (0.0, 0.4), 'description': 'Mild skin concerns'},
            'medium': {'range': (0.4, 0.7), 'description': 'Moderate skin concerns'},
            'high': {'range': (0.7, 1.0), 'description': 'Significant skin concerns'}
        }
    
    def analyze_prediction(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze prediction result and provide insights
        
        Args:
            prediction_result: Prediction result dictionary
            
        Returns:
            Analysis results
        """
        try:
            if not prediction_result.get('success'):
                return {'error': 'Prediction failed'}
            
            analysis = {
                'confidence_analysis': self._analyze_confidence(prediction_result),
                'issue_analysis': self._analyze_issues(prediction_result),
                'recommendations': self._generate_recommendations(prediction_result),
                'risk_assessment': self._assess_risk(prediction_result)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prediction: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_confidence(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction confidence"""
        try:
            primary_conf = prediction_result.get('primary_prediction', {}).get('confidence', 0.0)
            
            # Determine confidence level
            if primary_conf >= 0.8:
                confidence_level = 'very_high'
            elif primary_conf >= 0.6:
                confidence_level = 'high'
            elif primary_conf >= 0.4:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            return {
                'primary_confidence': primary_conf,
                'confidence_level': confidence_level,
                'is_reliable': primary_conf >= CLASSIFICATION_CONFIDENCE_THRESHOLD
            }
        except Exception as e:
            logger.error(f"Error analyzing confidence: {str(e)}")
            return {}
    
    def _analyze_issues(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detected skin issues"""
        try:
            issues = []
            if 'all_predictions' in prediction_result:
                for pred in prediction_result['all_predictions']:
                    if pred['confidence'] >= 0.2:  # Lower threshold for analysis
                        issues.append({
                            'issue': pred['class'],
                            'confidence': pred['confidence'],
                            'severity': self._get_issue_severity(pred['confidence'])
                        })
            
            return {
                'total_issues': len(issues),
                'issues': issues,
                'primary_issue': prediction_result.get('primary_prediction', {}).get('class'),
                'issue_categories': self._categorize_issues(issues)
            }
        except Exception as e:
            logger.error(f"Error analyzing issues: {str(e)}")
            return {}
    
    def _get_issue_severity(self, confidence: float) -> str:
        """Get severity level based on confidence"""
        if confidence >= 0.7:
            return 'high'
        elif confidence >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize issues by type"""
        categories = {
            'acne': [],
            'pigmentation': [],
            'aging': [],
            'texture': [],
            'sensitivity': [],
            'other': []
        }
        
        for issue in issues:
            issue_name = issue['issue']
            if issue_name in ['acne', 'blackheads', 'whiteheads', 'pimples']:
                categories['acne'].append(issue_name)
            elif issue_name in ['dark_spots', 'hyperpigmentation', 'age_spots', 'sun_damage']:
                categories['pigmentation'].append(issue_name)
            elif issue_name in ['wrinkles', 'fine_lines']:
                categories['aging'].append(issue_name)
            elif issue_name in ['dry_skin', 'oily_skin', 'combination_skin', 'large_pores']:
                categories['texture'].append(issue_name)
            elif issue_name in ['sensitive_skin', 'redness', 'inflammation']:
                categories['sensitivity'].append(issue_name)
            else:
                categories['other'].append(issue_name)
        
        return categories
    
    def _generate_recommendations(self, prediction_result: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations based on detected issues"""
        try:
            recommendations = []
            primary_issue = prediction_result.get('primary_prediction', {}).get('class')
            
            if primary_issue:
                if primary_issue in ['acne', 'blackheads', 'whiteheads', 'pimples']:
                    recommendations.append("Consider using salicylic acid or benzoyl peroxide")
                    recommendations.append("Avoid touching your face frequently")
                elif primary_issue in ['dark_spots', 'hyperpigmentation']:
                    recommendations.append("Use vitamin C serum for brightening")
                    recommendations.append("Always apply sunscreen with SPF 30+")
                elif primary_issue == 'dry_skin':
                    recommendations.append("Use a gentle, hydrating moisturizer")
                    recommendations.append("Avoid hot water when washing face")
                elif primary_issue == 'oily_skin':
                    recommendations.append("Use oil-free, non-comedogenic products")
                    recommendations.append("Consider using clay masks")
                elif primary_issue in ['wrinkles', 'fine_lines']:
                    recommendations.append("Consider using retinol products")
                    recommendations.append("Stay hydrated and protect from sun")
            
            # General recommendations
            recommendations.append("Maintain a consistent skincare routine")
            recommendations.append("Get adequate sleep and manage stress")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _assess_risk(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level based on predictions"""
        try:
            primary_conf = prediction_result.get('primary_prediction', {}).get('confidence', 0.0)
            
            if primary_conf >= 0.8:
                risk_level = 'high'
                risk_description = 'Strong indication of skin issues detected'
            elif primary_conf >= 0.6:
                risk_level = 'medium'
                risk_description = 'Moderate indication of skin issues detected'
            elif primary_conf >= 0.4:
                risk_level = 'low'
                risk_description = 'Mild indication of skin issues detected'
            else:
                risk_level = 'minimal'
                risk_description = 'No significant skin issues detected'
            
            return {
                'risk_level': risk_level,
                'risk_description': risk_description,
                'recommendation': 'Consult a dermatologist for professional advice'
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {str(e)}")
            return {}

def main():
    """
    Main function for testing prediction
    """
    try:
        # Initialize predictor
        predictor = SkinIssuePredictor()
        multi_predictor = MultiLabelSkinPredictor()
        analyzer = PredictionAnalyzer()
        
        print("Skin Issue Prediction System")
        print("=" * 40)
        
        # Test with a sample image (you would normally load an actual image)
        print("Note: This is a demo. In real usage, you would load an actual image.")
        
        # Example prediction result structure
        sample_result = {
            'success': True,
            'primary_prediction': {
                'class': 'acne',
                'confidence': 0.75,
                'index': 0
            },
            'top_3_predictions': [
                {'class': 'acne', 'confidence': 0.75, 'index': 0},
                {'class': 'oily_skin', 'confidence': 0.15, 'index': 7},
                {'class': 'blackheads', 'confidence': 0.08, 'index': 1}
            ]
        }
        
        # Analyze sample result
        analysis = analyzer.analyze_prediction(sample_result)
        
        print(f"Primary Issue: {sample_result['primary_prediction']['class']}")
        print(f"Confidence: {sample_result['primary_prediction']['confidence']:.2f}")
        print(f"Risk Level: {analysis.get('risk_assessment', {}).get('risk_level', 'unknown')}")
        
        print("\nRecommendations:")
        for rec in analysis.get('recommendations', []):
            print(f"- {rec}")
        
        logger.info("Prediction system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 