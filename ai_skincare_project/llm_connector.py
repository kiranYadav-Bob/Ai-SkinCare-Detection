"""
LLM Connector for AI Skincare Project
Connects to various LLM APIs for generating recommendations
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE,
    API_ENDPOINTS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMConnector:
    """
    Connects to various LLM APIs for generating skincare recommendations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.max_tokens = OPENAI_MAX_TOKENS
        self.temperature = OPENAI_TEMPERATURE
        self.endpoints = API_ENDPOINTS
        
        # Check if API key is available
        self.client_available = bool(self.api_key)
        if not self.client_available:
            logger.warning("No API key provided - using rule-based fallback")
    
    def generate_recommendations(self, 
                               skin_issues: List[str], 
                               image_analysis: Dict[str, Any],
                               user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate personalized skincare recommendations
        
        Args:
            skin_issues: List of detected skin issues
            image_analysis: Results from image analysis
            user_profile: Optional user profile information
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            if self.client_available:
                # Try to use OpenAI API
                return self._generate_openai_recommendations(skin_issues, image_analysis, user_profile)
            else:
                # Fallback to rule-based recommendations
                return self._generate_rule_based_recommendations(skin_issues, image_analysis, user_profile)
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            # Always fallback to rule-based
            return self._generate_rule_based_recommendations(skin_issues, image_analysis, user_profile)
    
    def _generate_openai_recommendations(self, 
                                       skin_issues: List[str], 
                                       image_analysis: Dict[str, Any],
                                       user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate recommendations using OpenAI API
        
        Args:
            skin_issues: List of detected skin issues
            image_analysis: Results from image analysis
            user_profile: Optional user profile information
            
        Returns:
            Dictionary containing OpenAI-generated recommendations
        """
        try:
            # Prepare prompt
            prompt = self._create_openai_prompt(skin_issues, image_analysis, user_profile)
            
            # Make API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert dermatologist specializing in skincare. Provide personalized, practical advice."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                self.endpoints["openai"],
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse the response
                recommendations = self._parse_openai_response(content)
                recommendations['source'] = 'openai'
                recommendations['success'] = True
                
                return recommendations
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                raise Exception(f"API call failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error with OpenAI API: {str(e)}")
            raise
    
    def _create_openai_prompt(self, 
                             skin_issues: List[str], 
                             image_analysis: Dict[str, Any],
                             user_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Create prompt for OpenAI API
        
        Args:
            skin_issues: List of detected skin issues
            image_analysis: Results from image analysis
            user_profile: Optional user profile information
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        As a dermatologist, analyze the following skin issues and provide personalized recommendations:
        
        Detected Skin Issues: {', '.join(skin_issues)}
        
        Image Analysis Results:
        - Skin tone: {image_analysis.get('skin_tone', 'Not detected')}
        - Texture: {image_analysis.get('texture', 'Not detected')}
        - Overall condition: {image_analysis.get('overall_condition', 'Not detected')}
        
        {f"User Profile: {json.dumps(user_profile, indent=2)}" if user_profile else ""}
        
        Please provide:
        1. Immediate treatment steps
        2. Product recommendations (specific ingredients and types)
        3. Lifestyle changes
        4. Prevention tips
        5. When to see a dermatologist
        
        Format your response as JSON with these keys:
        - treatment_steps: list of immediate actions
        - product_recommendations: list of product types/ingredients
        - lifestyle_changes: list of lifestyle modifications
        - prevention_tips: list of prevention strategies
        - dermatologist_consultation: when to seek professional help
        - severity_assessment: mild/moderate/severe
        - timeline: expected improvement timeline
        """
        
        return prompt
    
    def _parse_openai_response(self, content: str) -> Dict[str, Any]:
        """
        Parse OpenAI API response
        
        Args:
            content: Raw response content
            
        Returns:
            Parsed recommendations dictionary
        """
        try:
            # Try to extract JSON from response
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                
                parsed = json.loads(json_str)
                return parsed
            else:
                # Fallback parsing
                return self._fallback_parse(content)
                
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {str(e)}")
            return self._fallback_parse(content)
    
    def _fallback_parse(self, content: str) -> Dict[str, Any]:
        """
        Fallback parsing for non-JSON responses
        
        Args:
            content: Raw response content
            
        Returns:
            Parsed recommendations dictionary
        """
        # Simple text parsing as fallback
        recommendations = {
            'treatment_steps': ['Follow the recommendations provided'],
            'product_recommendations': ['Consult with a dermatologist for specific products'],
            'lifestyle_changes': ['Maintain good skincare habits'],
            'prevention_tips': ['Use sunscreen daily'],
            'dermatologist_consultation': 'If issues persist or worsen',
            'severity_assessment': 'moderate',
            'timeline': '4-6 weeks with consistent care',
            'raw_response': content
        }
        
        return recommendations
    
    def _generate_rule_based_recommendations(self, 
                                           skin_issues: List[str], 
                                           image_analysis: Dict[str, Any],
                                           user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate rule-based recommendations as fallback
        
        Args:
            skin_issues: List of detected skin issues
            image_analysis: Results from image analysis
            user_profile: Optional user profile information
            
        Returns:
            Dictionary containing rule-based recommendations
        """
        try:
            recommendations = {
                'treatment_steps': [],
                'product_recommendations': [],
                'lifestyle_changes': [],
                'prevention_tips': [],
                'dermatologist_consultation': 'If issues persist or worsen',
                'severity_assessment': 'moderate',
                'timeline': '4-6 weeks with consistent care',
                'source': 'rule_based'
            }
            
            # Generate recommendations based on detected issues
            for issue in skin_issues:
                issue_recs = self._get_issue_specific_recommendations(issue)
                
                recommendations['treatment_steps'].extend(issue_recs.get('treatment', []))
                recommendations['product_recommendations'].extend(issue_recs.get('products', []))
                recommendations['lifestyle_changes'].extend(issue_recs.get('lifestyle', []))
                recommendations['prevention_tips'].extend(issue_recs.get('prevention', []))
            
            # Remove duplicates
            for key in ['treatment_steps', 'product_recommendations', 'lifestyle_changes', 'prevention_tips']:
                recommendations[key] = list(set(recommendations[key]))
            
            recommendations['success'] = True
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating rule-based recommendations: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'source': 'rule_based_fallback'
            }
    
    def _get_issue_specific_recommendations(self, issue: str) -> Dict[str, List[str]]:
        """
        Get issue-specific recommendations
        
        Args:
            issue: Skin issue name
            
        Returns:
            Dictionary with recommendations for the specific issue
        """
        issue_recs = {
            'acne': {
                'treatment': [
                    'Gently cleanse twice daily with non-comedogenic cleanser',
                    'Apply benzoyl peroxide or salicylic acid treatment',
                    'Avoid picking or popping pimples',
                    'Use oil-free moisturizer'
                ],
                'products': [
                    'Benzoyl peroxide 2.5-5%',
                    'Salicylic acid cleanser',
                    'Non-comedogenic moisturizer',
                    'Oil-free sunscreen'
                ],
                'lifestyle': [
                    'Keep hands away from face',
                    'Change pillowcase regularly',
                    'Avoid touching face throughout the day'
                ],
                'prevention': [
                    'Maintain consistent skincare routine',
                    'Avoid heavy makeup',
                    'Clean makeup brushes regularly'
                ]
            },
            'dry_skin': {
                'treatment': [
                    'Use gentle, fragrance-free cleanser',
                    'Apply thick moisturizer immediately after bathing',
                    'Use humidifier in dry environments',
                    'Limit hot showers'
                ],
                'products': [
                    'Hyaluronic acid serum',
                    'Ceramide moisturizer',
                    'Gentle cleanser',
                    'Occlusive ointment'
                ],
                'lifestyle': [
                    'Drink plenty of water',
                    'Avoid harsh soaps',
                    'Use lukewarm water for washing'
                ],
                'prevention': [
                    'Moisturize regularly',
                    'Protect from harsh weather',
                    'Use gentle skincare products'
                ]
            },
            'oily_skin': {
                'treatment': [
                    'Cleanse twice daily with gentle cleanser',
                    'Use oil-free moisturizer',
                    'Apply clay mask weekly',
                    'Blot excess oil throughout the day'
                ],
                'products': [
                    'Oil-free cleanser',
                    'Salicylic acid toner',
                    'Oil-free moisturizer',
                    'Clay mask'
                ],
                'lifestyle': [
                    'Don\'t skip moisturizer',
                    'Avoid over-cleansing',
                    'Use blotting papers'
                ],
                'prevention': [
                    'Maintain balanced skincare routine',
                    'Avoid harsh products',
                    'Keep skin hydrated'
                ]
            }
        }
        
        # Return recommendations for the specific issue, or general ones if not found
        return issue_recs.get(issue, {
            'treatment': ['Consult with a dermatologist for personalized treatment'],
            'products': ['Use gentle, fragrance-free products'],
            'lifestyle': ['Maintain good skincare habits'],
            'prevention': ['Protect skin from sun damage']
        })

def main():
    """Test the LLM connector"""
    try:
        connector = LLMConnector()
        print("LLM Connector initialized successfully!")
        
        # Test with sample data
        test_issues = ['acne', 'dry_skin']
        test_analysis = {'skin_tone': 'medium', 'texture': 'rough', 'overall_condition': 'fair'}
        
        recommendations = connector.generate_recommendations(test_issues, test_analysis)
        
        print(f"Recommendations generated: {recommendations['success']}")
        print(f"Source: {recommendations.get('source', 'unknown')}")
        print(f"Treatment steps: {len(recommendations.get('treatment_steps', []))}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
