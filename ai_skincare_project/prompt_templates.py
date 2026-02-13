"""
Prompt Templates for AI Skincare Project
Contains structured prompts for different AI interactions
"""

from typing import Dict, List, Any
class PromptTemplateManager:
    """
    Manages available prompt templates and provides easy retrieval.
    """
    def __init__(self):
        self.templates = {
            'skin_analysis': PromptTemplates.skin_analysis_prompt,
            'treatment_recommendations': PromptTemplates.treatment_recommendations_prompt,
            'product_recommendations': PromptTemplates.product_recommendations_prompt,
            'lifestyle_recommendations': PromptTemplates.lifestyle_recommendations_prompt,
            'emergency_consultation': PromptTemplates.emergency_consultation_prompt,
            'follow_up': PromptTemplates.follow_up_prompt
        }

    def get_template(self, template_type: str, **kwargs) -> str:
        """
        Retrieve and format a specific prompt template.
        """
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        return self.templates[template_type](**kwargs)

    def list_templates(self):
        """Return a list of available template types."""
        return list(self.templates.keys())

class PromptTemplates:
    """
    Collection of prompt templates for various AI interactions
    """
    
    @staticmethod
    def skin_analysis_prompt(skin_issues: List[str], image_details: Dict[str, Any]) -> str:
        """
        Generate prompt for skin analysis
        
        Args:
            skin_issues: List of detected skin issues
            image_details: Details about the analyzed image
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        Analyze the following skin image and provide detailed insights:
        
        Detected Skin Issues: {', '.join(skin_issues) if skin_issues else 'None detected'}
        
        Image Details:
        - Resolution: {image_details.get('resolution', 'Unknown')}
        - Lighting: {image_details.get('lighting', 'Unknown')}
        - Image Quality: {image_details.get('quality', 'Unknown')}
        
        Please provide:
        1. Detailed analysis of each detected issue
        2. Severity assessment (mild/moderate/severe)
        3. Potential causes
        4. Immediate concerns to address
        5. Overall skin health assessment
        
        Format your response as a structured analysis with clear sections.
        """
        return prompt
    
    @staticmethod
    def treatment_recommendations_prompt(skin_issues: List[str], severity: str) -> str:
        """
        Generate prompt for treatment recommendations
        
        Args:
            skin_issues: List of detected skin issues
            severity: Overall severity assessment
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        As a dermatologist, provide comprehensive treatment recommendations for:
        
        Skin Issues: {', '.join(skin_issues)}
        Overall Severity: {severity}
        
        Please provide:
        
        1. IMMEDIATE TREATMENT STEPS:
           - First 24-48 hours actions
           - What to avoid
           - Emergency signs to watch for
        
        2. DAILY TREATMENT ROUTINE:
           - Morning routine
           - Evening routine
           - Weekly treatments
        
        3. PRODUCT RECOMMENDATIONS:
           - Specific ingredients to look for
           - Product types (cleanser, moisturizer, treatment)
           - Application order and frequency
        
        4. LIFESTYLE MODIFICATIONS:
           - Diet changes
           - Sleep recommendations
           - Stress management
           - Environmental factors
        
        5. PREVENTION STRATEGIES:
           - Long-term habits
           - Seasonal adjustments
           - Maintenance routine
        
        6. TIMELINE:
           - Expected improvement timeline
           - When to expect results
           - Follow-up schedule
        
        7. PROFESSIONAL CONSULTATION:
           - When to see a dermatologist
           - What to bring to appointment
           - Questions to ask
        
        Provide practical, actionable advice suitable for home care.
        """
        return prompt
    
    @staticmethod
    def product_recommendations_prompt(skin_issues: List[str], skin_type: str, budget: str = "moderate") -> str:
        """
        Generate prompt for product recommendations
        
        Args:
            skin_issues: List of detected skin issues
            skin_type: Type of skin (dry, oily, combination, sensitive)
            budget: Budget category (budget, moderate, premium)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        Provide specific product recommendations for:
        
        Skin Issues: {', '.join(skin_issues)}
        Skin Type: {skin_type}
        Budget: {budget}
        
        Please recommend:
        
        1. CLEANSER:
           - Specific product recommendations
           - Key ingredients to look for
           - Application method
           - Frequency of use
        
        2. TREATMENT PRODUCTS:
           - Active ingredients for each issue
           - Product types (serum, cream, gel)
           - Application order
           - When to use each
        
        3. MOISTURIZER:
           - Texture recommendations
           - Key ingredients
           - Application timing
           - Layering with other products
        
        4. SUNSCREEN:
           - SPF level
           - Texture and finish
           - Reapplication schedule
           - Compatibility with other products
        
        5. WEEKLY TREATMENTS:
           - Masks
           - Exfoliants
           - Special treatments
        
        6. AVOID:
           - Ingredients to avoid
           - Product combinations to skip
           - Common mistakes
        
        Include specific brand recommendations within the budget range.
        """
        return prompt
    
    @staticmethod
    def lifestyle_recommendations_prompt(skin_issues: List[str], age_group: str = "adult") -> str:
        """
        Generate prompt for lifestyle recommendations
        
        Args:
            skin_issues: List of detected skin issues
            age_group: Age group (teen, adult, senior)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        Provide lifestyle recommendations for improving skin health:
        
        Skin Issues: {', '.join(skin_issues)}
        Age Group: {age_group}
        
        Please cover:
        
        1. DIET RECOMMENDATIONS:
           - Foods to include
           - Foods to avoid
           - Supplements
           - Hydration tips
        
        2. SLEEP OPTIMIZATION:
           - Sleep duration
           - Sleep position
           - Pillowcase hygiene
           - Bedroom environment
        
        3. STRESS MANAGEMENT:
           - Stress-skin connection
           - Relaxation techniques
           - Exercise recommendations
           - Mindfulness practices
        
        4. ENVIRONMENTAL PROTECTION:
           - Sun protection
           - Pollution protection
           - Climate considerations
           - Indoor air quality
        
        5. DAILY HABITS:
           - Morning routine
           - Evening routine
           - Weekly habits
           - Monthly maintenance
        
        6. AVOIDANCE BEHAVIORS:
           - Habits to stop
           - Triggers to avoid
           - Environmental factors
           - Product misuse
        
        7. MONITORING AND TRACKING:
           - Progress indicators
           - Warning signs
           - Journal suggestions
           - Photo documentation
        
        Provide age-appropriate, practical advice.
        """
        return prompt
    
    @staticmethod
    def emergency_consultation_prompt(skin_issues: List[str], symptoms: List[str]) -> str:
        """
        Generate prompt for emergency consultation guidance
        
        Args:
            skin_issues: List of detected skin issues
            symptoms: Current symptoms
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        Assess the urgency of seeking medical attention for:
        
        Skin Issues: {', '.join(skin_issues)}
        Current Symptoms: {', '.join(symptoms)}
        
        Please evaluate:
        
        1. URGENCY ASSESSMENT:
           - Immediate (within hours)
           - Urgent (within 24-48 hours)
           - Routine (within 1-2 weeks)
           - Self-care only
        
        2. EMERGENCY SIGNS:
           - Red flags to watch for
           - Symptoms requiring immediate attention
           - Pain or discomfort levels
           - Systemic symptoms
        
        3. IMMEDIATE ACTIONS:
           - What to do right now
           - Home remedies to try
           - What to avoid
           - Pain management
        
        4. PREPARATION FOR APPOINTMENT:
           - Information to gather
           - Photos to take
           - Questions to prepare
           - What to bring
        
        5. SPECIALIST RECOMMENDATIONS:
           - Type of doctor to see
           - When to seek second opinion
           - Telemedicine options
           - Emergency room vs. urgent care
        
        6. MONITORING:
           - What to track
           - Warning signs
           - Improvement indicators
           - When to follow up
        
        Prioritize patient safety and provide clear guidance on urgency.
        """
        return prompt
    
    @staticmethod
    def follow_up_prompt(previous_issues: List[str], current_status: str, timeline: str) -> str:
        """
        Generate prompt for follow-up recommendations
        
        Args:
            previous_issues: Previously identified skin issues
            current_status: Current skin condition
            timeline: Time since last analysis
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
        Provide follow-up recommendations based on:
        
        Previous Issues: {', '.join(previous_issues)}
        Current Status: {current_status}
        Time Since Last Analysis: {timeline}
        
        Please assess:
        
        1. PROGRESS EVALUATION:
           - Improvement assessment
           - Remaining issues
           - New concerns
           - Overall progress
        
        2. ADJUSTED TREATMENT PLAN:
           - What's working
           - What needs adjustment
           - New treatments to try
           - Discontinue ineffective treatments
        
        3. MAINTENANCE ROUTINE:
           - Long-term habits
           - Prevention strategies
           - Seasonal adjustments
           - Ongoing care
        
        4. NEXT STEPS:
           - Immediate actions
           - Short-term goals (1-2 weeks)
           - Medium-term goals (1-2 months)
           - Long-term goals (3-6 months)
        
        5. MONITORING PLAN:
           - What to track
           - Progress indicators
           - Warning signs
           - Follow-up schedule
        
        6. LIFESTYLE ADJUSTMENTS:
           - Diet modifications
           - Stress management
           - Sleep optimization
           - Environmental factors
        
        7. PROFESSIONAL FOLLOW-UP:
           - When to see doctor
           - What to discuss
           - Tests to request
           - Specialist referrals
        
        Focus on continued improvement and long-term skin health.
        """
        return prompt
    
    
def create_prompt_from_analysis(analysis_results: dict) -> str:
    """
    Create a default prompt string from skin analysis results.
    """
    prompt = "Skin Analysis Results:\n"
    for key, value in analysis_results.items():
        prompt += f"- {key}: {value}\n"
    prompt += "\nPlease provide detailed skincare recommendations."
    return prompt


def get_prompt_template(template_type: str, **kwargs) -> str:
    """
    Get a specific prompt template
    
    Args:
        template_type: Type of template to retrieve
        **kwargs: Arguments for the template
        
    Returns:
        Formatted prompt string
    """
    templates = {
        'skin_analysis': PromptTemplates.skin_analysis_prompt,
        'treatment_recommendations': PromptTemplates.treatment_recommendations_prompt,
        'product_recommendations': PromptTemplates.product_recommendations_prompt,
        'lifestyle_recommendations': PromptTemplates.lifestyle_recommendations_prompt,
        'emergency_consultation': PromptTemplates.emergency_consultation_prompt,
        'follow_up': PromptTemplates.follow_up_prompt
    }
    
    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}")
    
    return templates[template_type](**kwargs)

def main():
    """Test the prompt templates"""
    try:
        # Test skin analysis prompt
        skin_issues = ['acne', 'dry_skin']
        image_details = {'resolution': '1920x1080', 'lighting': 'natural', 'quality': 'high'}
        
        prompt = PromptTemplates.skin_analysis_prompt(skin_issues, image_details)
        print("Skin Analysis Prompt:")
        print(prompt)
        print("\n" + "="*50 + "\n")
        
        # Test treatment recommendations prompt
        treatment_prompt = PromptTemplates.treatment_recommendations_prompt(skin_issues, 'moderate')
        print("Treatment Recommendations Prompt:")
        print(treatment_prompt)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")



if __name__ == "__main__":
    main()
