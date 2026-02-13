"""
AI Skincare Dashboard
Complete Streamlit dashboard for AI-powered skincare analysis and recommendations
"""

import streamlit as st
import pandas as pd
import logging
import os
from PIL import Image

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from image_utils import *
from image_upload import ImageUploadHandler
from face_crop import FaceCropProcessor
from predict_skin_issue import MultiLabelSkinPredictor
from agentic_derm_flow import create_agentic_flow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI/ML Skincare Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class SkincareDashboard:
    """Main dashboard class for AI skincare analysis"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.upload_handler = ImageUploadHandler()
        self.face_processor = FaceCropProcessor()
        self.predictor = MultiLabelSkinPredictor()
        
        # Initialize session state
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        
    def run(self):
        """Run the main dashboard"""
        # Header with reset button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown('<h1 class="main-header">🤖 AI/ML Skincare Analysis</h1>', unsafe_allow_html=True)
            st.markdown("### Data Science & Machine Learning powered skin analysis")
        
        with col2:
            st.markdown("")  # Spacing
            st.markdown("")  # Spacing
            if st.button("🔄 Reset Analysis", type="secondary", use_container_width=True):
                self._reset_analysis()
                st.rerun()
        
        st.markdown("---")
        
        # Main content based on current step
        if st.session_state.current_step == 1:
            self._upload_section()
        elif st.session_state.current_step == 2:
            self._analysis_section()
        elif st.session_state.current_step == 3:
            self._recommendations_section()
        
        # Step indicator
        self._show_step_indicator()
    
    def _reset_analysis(self):
        """Reset all analysis data"""
        st.session_state.current_step = 1
        st.session_state.analysis_results = {}
        st.session_state.uploaded_image = None
        st.success("✅ Analysis reset successfully!")
    
    def _upload_section(self):
        """Image upload section"""
        st.header("📷 Step 1: Upload Your Image")
        
        st.markdown("""
        **Upload a clear photo of your face for AI/ML analysis**
        
        **Requirements:**
        - Clear, well-lit face photo
        - File formats: PNG, JPG, JPEG
        - Maximum file size: 10MB
        """)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear photo of your face for skin analysis"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing uploaded image..."):
                result = self.upload_handler.process_uploaded_file(uploaded_file)
                
                if result and result.get('success'):
                    st.session_state.uploaded_image = result
                    st.session_state.current_step = 2
                    st.success("✅ Image uploaded and processed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to process image. Please try again with a different image.")
        
        # Show upload tips
        with st.expander("💡 Upload Tips"):
            st.markdown("""
            **For best AI/ML analysis results:**
            
            ✅ **Good practices:**
            - Use natural lighting
            - Face should be clearly visible
            - Remove glasses/accessories if possible
            - High resolution images (at least 300x300 pixels)
            
            ❌ **Avoid:**
            - Blurry or dark images
            - Side profile shots
            - Images with multiple faces
            - Heavily filtered photos
            """)
    
    def _analysis_section(self):
        """Skin issue analysis section"""
        st.header("🔬 Step 2: Skin Issue Analysis")
        
        if not st.session_state.uploaded_image:
            st.error("No image uploaded. Please go back to Step 1.")
            return
        
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(st.session_state.uploaded_image['pil_image'], caption="Original Image", use_container_width=True)
            
            # Image info
            st.markdown(f"**Image Details:**")
            st.markdown(f"- Size: {st.session_state.uploaded_image['image_size']}")
            st.markdown(f"- Format: {st.session_state.uploaded_image['image_format']}")
            st.markdown(f"- File size: {st.session_state.uploaded_image['file_size']}")
        
        with col2:
            st.subheader("🔬 Analysis Progress")
            
            # Step 1: Face Detection
            with st.spinner("Detecting faces..."):
                face_result = self.face_processor.process_image(st.session_state.uploaded_image['pil_image'])
                if face_result['success']:
                    st.success("✅ Face detected successfully!")
                    # Display the first cropped face if available
                    if face_result.get('cropped_faces'):
                        st.image(face_result['cropped_faces'][0], caption="Detected Face", use_container_width=True)
                else:
                    st.warning("⚠️ Face detection limited - proceeding with full image analysis")
            
            # Step 2: Skin Issue Prediction
            with st.spinner("Analyzing skin issues..."):
                try:
                    # Use the first cropped face if available, otherwise use original
                    if face_result.get('success') and face_result.get('cropped_faces'):
                        analysis_image = face_result['cropped_faces'][0]
                    else:
                        analysis_image = st.session_state.uploaded_image['pil_image']
                    predictions = self.predictor.predict_multi_label(analysis_image)
                    
                    if predictions and len(predictions) > 0:
                        st.success("✅ Skin analysis completed!")
                        st.session_state.analysis_results = {
                            'predictions': predictions,
                            'face_result': face_result,
                            'original_image': st.session_state.uploaded_image
                        }
                        st.session_state.current_step = 3
                        st.rerun()
                    else:
                        st.error("❌ No skin issues detected. Please try with a different image.")
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")
        
        # Navigation
        if st.button("← Back to Upload"):
            st.session_state.current_step = 1
            st.rerun()
    
    def _recommendations_section(self):
        """AI recommendations section"""
        st.header("🤖 Step 3: AI-Powered Recommendations")
        
        if not st.session_state.analysis_results:
            st.error("Analysis not completed.")
            return
        
        # Display analysis results
        self._display_analysis_results()
        
        # Generate recommendations
        if st.button("🚀 Generate AI/ML Recommendations", type="primary"):
            with st.spinner("Generating personalized recommendations..."):
                try:
                    # Get detected issues for recommendations
                    detected_issues = []
                    predictions = st.session_state.analysis_results.get('predictions')
                    if predictions:
                        if isinstance(predictions, dict) and 'detected_issues' in predictions:
                            # Multi-label format
                            detected_issues = predictions['detected_issues']
                        elif isinstance(predictions, list):
                            # List format
                            for pred in predictions:
                                if isinstance(pred, dict):
                                    if pred.get('confidence', 0) > 0.3:
                                        detected_issues.append(pred.get('issue', pred.get('class', 'Unknown')))
                                elif isinstance(pred, str):
                                    detected_issues.append(pred)
                    
                    if detected_issues:
                        # Generate recommendations using agentic flow
                        agentic_flow = create_agentic_flow(None)
                        # Create skin_analysis dictionary for the agentic flow
                        skin_analysis = {
                            'detected_issues': detected_issues,
                            'analysis_type': 'multi_label',
                            'confidence_threshold': 0.3
                        }
                        recommendations = agentic_flow.run_complete_analysis(
                            skin_analysis=skin_analysis,
                            user_profile=None  # No user profile feature
                        )
                        
                        if recommendations:
                            st.session_state.analysis_results['recommendations'] = recommendations
                            st.success("✅ Recommendations generated successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to generate recommendations.")
                    else:
                        st.warning("⚠️ No significant skin issues detected for recommendations.")
                        
                except Exception as e:
                    st.error(f"❌ Recommendation generation failed: {str(e)}")
                    logger.error(f"Recommendation error: {e}")
        
        # Display recommendations if available
        if st.session_state.analysis_results.get('recommendations'):
            self._display_recommendations()
        
        # Navigation
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Analysis"):
                st.session_state.current_step = 2
                st.rerun()
        
        with col2:
            if st.button("🔄 New Analysis"):
                st.session_state.current_step = 1
                st.session_state.analysis_results = {}
                st.rerun()
    
    def _display_analysis_results(self):
        """Display skin analysis results"""
        st.subheader("📊 Analysis Results")
        
        if not st.session_state.analysis_results.get('predictions'):
            st.info("ℹ️ No specific skin issues detected.")
            return
        
        # Create DataFrame for detected issues
        predictions = st.session_state.analysis_results['predictions']
        detected_issues = []
        issue_to_conf = {}
        
        # Check if predictions is a multi-label result
        if isinstance(predictions, dict) and 'detected_issues' in predictions:
            # Multi-label format
            detected_issue_names = predictions['detected_issues']
            # Build confidence map from category_results if available
            category_results = predictions.get('category_results', {})
            if isinstance(category_results, dict):
                for category_info in category_results.values():
                    for pred in category_info.get('predictions', []) or []:
                        issue_to_conf[pred.get('class', '')] = float(pred.get('confidence', 0.0))
            for issue_name in detected_issue_names:
                conf = issue_to_conf.get(issue_name, None)
                detected_issues.append({
                    'Skin Issue': issue_name.replace('_', ' ').title(),
                    'Confidence': f"{conf:.1%}" if isinstance(conf, float) else '—',
                    'Severity': 'High' if (isinstance(conf, float) and conf > 0.7) else 'Medium' if (isinstance(conf, float) and conf > 0.5) else 'Low' if isinstance(conf, float) else 'Medium'
                })
        elif isinstance(predictions, list):
            # List format - check if it's list of dicts or strings
            for pred in predictions:
                if isinstance(pred, dict):
                    if pred.get('confidence', 0) > 0.3:  # Use default threshold
                        issue = pred.get('issue', pred.get('class', 'Unknown'))
                        confidence = pred.get('confidence', 0)
                        # Format issue name for display
                        display_issue = issue.replace('_', ' ').title()
                        detected_issues.append({
                            'Skin Issue': display_issue,
                            'Confidence': f"{confidence:.1%}",
                            'Severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'
                        })
                elif isinstance(pred, str):
                    detected_issues.append({
                        'Skin Issue': pred.replace('_', ' ').title(),
                        'Confidence': 'High',
                        'Severity': 'Medium'
                    })
        
        if detected_issues:
            df = pd.DataFrame(detected_issues)
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            st.markdown(f"**Total Issues Detected:** {len(detected_issues)}")
            high_severity = len([i for i in detected_issues if i['Severity'] == 'High'])
            if high_severity > 0:
                st.warning(f"⚠️ {high_severity} high-severity issues detected")
            
            # Show specific detected problems
            st.markdown("### 🔍 Detected Problems:")
            for i, issue in enumerate(detected_issues, 1):
                issue_name = issue['Skin Issue']
                confidence = issue['Confidence']
                severity = issue['Severity']
                
                # Color coding based on severity
                if severity == 'High':
                    st.markdown(f"**{i}. {issue_name}** 🔴 (High Priority)")
                elif severity == 'Medium':
                    st.markdown(f"**{i}. {issue_name}** 🟡 (Medium Priority)")
                else:
                    st.markdown(f"**{i}. {issue_name}** 🟢 (Low Priority)")
                
                st.markdown(f"   - Confidence: {confidence}")
                st.markdown(f"   - Severity: {severity}")
                st.markdown("---")
        else:
            st.info("ℹ️ No significant skin issues detected above confidence threshold.")
    
    def _display_recommendations(self):
        """Display AI/ML recommendations"""
        st.subheader("🎯 Personalized Recommendations")
        
        recommendations = st.session_state.analysis_results['recommendations']
        
        # Display Detected Problems
        if recommendations.get('detected_problems'):
            with st.expander("🔍 Detected Skin Problems", expanded=True):
                detected_problems = recommendations['detected_problems']
                st.markdown(f"**Total Problems Detected:** {detected_problems.get('total_problems', 0)}")
                st.markdown(f"**Primary Concern:** {detected_problems.get('primary_concern', 'None')}")
                
                for i, problem in enumerate(detected_problems.get('problems', []), 1):
                    st.markdown(f"### {i}. {problem['name']}")
                    st.markdown(f"**Description:** {problem['description']}")
                    st.markdown(f"**Severity:** {problem['severity']}")
                    st.markdown("**Common Causes:**")
                    for cause in problem.get('common_causes', []):
                        st.markdown(f"- {cause}")
                    st.markdown("---")
        
        # Treatment Plan
        if recommendations.get('treatment_plan'):
            with st.expander("💊 Treatment Plan", expanded=True):
                st.markdown(recommendations['treatment_plan'])
        
        # Product Recommendations
        if recommendations.get('product_recommendations'):
            with st.expander("🛍️ Product Recommendations", expanded=True):
                product_recs = recommendations['product_recommendations']
                if isinstance(product_recs, list):
                    for product in product_recs:
                        st.markdown(f"- {product}")
                elif isinstance(product_recs, dict):
                    for category, products in product_recs.items():
                        st.markdown(f"**{category}:**")
                        for product in products:
                            st.markdown(f"  - {product}")
                else:
                    st.markdown(str(product_recs))
        
        # Lifestyle Recommendations
        if recommendations.get('lifestyle_recommendations'):
            with st.expander("🌱 Lifestyle & Prevention", expanded=True):
                lifestyle_recs = recommendations['lifestyle_recommendations']
                if isinstance(lifestyle_recs, list):
                    for tip in lifestyle_recs:
                        st.markdown(f"- {tip}")
                else:
                    st.markdown(str(lifestyle_recs))
        
        # Priority Actions
        if recommendations.get('priority_actions'):
            with st.expander("⚡ Priority Actions", expanded=True):
                priority_actions = recommendations['priority_actions']
                if isinstance(priority_actions, list):
                    for i, action in enumerate(priority_actions, 1):
                        st.markdown(f"{i}. {action}")
                else:
                    st.markdown(str(priority_actions))
        
        # Timeline
        if recommendations.get('timeline'):
            with st.expander("📅 Expected Timeline", expanded=True):
                st.markdown(recommendations['timeline'])
        
        # Follow-up Schedule
        if recommendations.get('follow_up_schedule'):
            with st.expander("📋 Follow-up Schedule", expanded=True):
                st.markdown(recommendations['follow_up_schedule'])
    
    def _show_step_indicator(self):
        """Show current step indicator"""
        steps = ["Upload", "Analysis", "Recommendations"]
        current_step = st.session_state.current_step
        
        st.markdown("---")
        cols = st.columns(len(steps))
        
        for i, (col, step) in enumerate(zip(cols, steps), 1):
            if i < current_step:
                col.markdown(f"✅ {step}")
            elif i == current_step:
                col.markdown(f"🔄 {step}")
            else:
                col.markdown(f"⏳ {step}")


def main():
    """Main function to run the dashboard"""
    try:
        dashboard = SkincareDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")


if __name__ == "__main__":
    main()
