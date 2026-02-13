"""
Agentic Dermatologist Flow
Multi-step, expert-like recommendation system with iterative analysis and personalized treatment planning
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Import from other modules
from llm_connector import LLMConnector
from prompt_templates import PromptTemplateManager, create_prompt_from_analysis

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisStep(Enum):
    """Enumeration of analysis steps in the agentic flow"""
    INITIAL_ASSESSMENT = "initial_assessment"
    DETAILED_ANALYSIS = "detailed_analysis"
    TREATMENT_PLANNING = "treatment_planning"
    PRODUCT_SELECTION = "product_selection"
    LIFESTYLE_INTEGRATION = "lifestyle_integration"
    FOLLOW_UP_PLANNING = "follow_up_planning"
    FINAL_RECOMMENDATION = "final_recommendation"

class SeverityLevel(Enum):
    """Enumeration of severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    step: AnalysisStep
    success: bool
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    notes: Optional[str] = None
    requires_follow_up: bool = False

@dataclass
class TreatmentPlan:
    """Data class for treatment plans"""
    plan_id: str
    user_id: str
    created_at: datetime
    skin_analysis: Dict[str, Any]
    user_profile: Dict[str, Any]
    treatment_steps: List[Dict[str, Any]]
    product_recommendations: Dict[str, List[str]]
    lifestyle_recommendations: List[str]
    follow_up_schedule: Dict[str, Any]
    priority_actions: List[str]
    risk_factors: List[str]
    expected_timeline: str
    success_metrics: Dict[str, Any]

class AgenticDermatologistFlow:
    """
    Multi-step, expert-like dermatologist recommendation system
    Implements iterative analysis and personalized treatment planning
    """
    
    def __init__(self, llm_connector: Optional[LLMConnector] = None):
        """
        Initialize the agentic dermatologist flow
        
        Args:
            llm_connector: LLM connector instance
        """
        self.llm_connector = llm_connector or LLMConnector()
        self.template_manager = PromptTemplateManager()
        self.analysis_history: List[AnalysisResult] = []
        self.current_plan: Optional[TreatmentPlan] = None
        
    def run_complete_analysis(self, 
                            skin_analysis: Dict[str, Any],
                            user_profile: Optional[Dict[str, Any]] = None,
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete agentic dermatologist analysis
        
        Args:
            skin_analysis: Results from skin issue detection
            user_profile: User profile information
            session_id: Session identifier
            
        Returns:
            Complete analysis and treatment plan
        """
        try:
            logger.info(f"Starting complete agentic analysis for session: {session_id}")
            
            # Check if we have LLM capabilities
            if not self.llm_connector or not self.llm_connector.client:
                logger.info("No LLM available, using rule-based recommendations")
                return self._generate_rule_based_recommendations(skin_analysis, user_profile, session_id)
            
            # Generate session ID if not provided
            session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Step 1: Initial Assessment
            initial_result = self._run_initial_assessment(skin_analysis, None) 
            self.analysis_history.append(initial_result)
            
            if not initial_result.success:
                return self._create_error_response("Initial assessment failed", initial_result.notes)
            
            # Step 2: Detailed Analysis
            detailed_result = self._run_detailed_analysis(skin_analysis, None, initial_result.data)
            self.analysis_history.append(detailed_result)
            
            # Step 3: Treatment Planning
            treatment_result = self._run_treatment_planning(skin_analysis, None, detailed_result.data) 
            self.analysis_history.append(treatment_result)
            
            # Step 4: Product Selection
            product_result = self._run_product_selection(treatment_result.data, None) 
            self.analysis_history.append(product_result)
            
            # Step 5: Lifestyle Integration
            lifestyle_result = self._run_lifestyle_integration(treatment_result.data, None) 
            self.analysis_history.append(lifestyle_result)
            
            # Step 6: Follow-up Planning
            follow_up_result = self._run_follow_up_planning(treatment_result.data, None) 
            self.analysis_history.append(follow_up_result)
            
            # Step 7: Final Recommendation
            final_result = self._create_final_recommendation(
                skin_analysis, None, session_id
            )
            
            # Create comprehensive response
            return {
                'success': True,
                'treatment_plan': asdict(final_result),
                'analysis_summary': self._create_analysis_summary(),
                'confidence_score': self._calculate_overall_confidence(),
                'risk_assessment': self._assess_risks(final_result),
                'next_actions': self._get_next_actions(final_result),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            # Fallback to rule-based recommendations
            return self._generate_rule_based_recommendations(skin_analysis, user_profile, session_id)
    
    def _run_initial_assessment(self, 
                               skin_analysis: Dict[str, Any],
                               user_profile: Optional[Dict[str, Any]]) -> AnalysisResult:
        """
        Run initial assessment of skin condition
        
        Args:
            skin_analysis: Skin analysis results
            user_profile: User profile information
            
        Returns:
            Initial assessment result
        """
        try:
            # Create initial assessment prompt
            prompt = self._create_initial_assessment_prompt(skin_analysis, user_profile)
            
            # Get LLM response
            response = self.llm_connector.generate_recommendation(
                skin_analysis, user_profile, prompt
            )
            
            if not response.get('success'):
                return AnalysisResult(
                    step=AnalysisStep.INITIAL_ASSESSMENT,
                    success=False,
                    data={},
                    confidence=0.0,
                    timestamp=datetime.now(),
                    notes=response.get('error', 'LLM response failed')
                )
            
            # Parse and structure the response
            assessment_data = self._parse_initial_assessment(response['recommendation'])
            
            return AnalysisResult(
                step=AnalysisStep.INITIAL_ASSESSMENT,
                success=True,
                data=assessment_data,
                confidence=assessment_data.get('confidence', 0.7),
                timestamp=datetime.now(),
                notes="Initial assessment completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in initial assessment: {str(e)}")
            return AnalysisResult(
                step=AnalysisStep.INITIAL_ASSESSMENT,
                success=False,
                data={},
                confidence=0.0,
                timestamp=datetime.now(),
                notes=str(e)
            )
    
    def _run_detailed_analysis(self, 
                              skin_analysis: Dict[str, Any],
                              user_profile: Optional[Dict[str, Any]],
                              initial_data: Dict[str, Any]) -> AnalysisResult:
        """
        Run detailed analysis based on initial assessment
        
        Args:
            skin_analysis: Skin analysis results
            user_profile: User profile information
            initial_data: Data from initial assessment
            
        Returns:
            Detailed analysis result
        """
        try:
            # Create detailed analysis prompt
            prompt = self._create_detailed_analysis_prompt(skin_analysis, user_profile, initial_data)
            
            # Get LLM response
            response = self.llm_connector.generate_recommendation(
                skin_analysis, user_profile, prompt
            )
            
            if not response.get('success'):
                return AnalysisResult(
                    step=AnalysisStep.DETAILED_ANALYSIS,
                    success=False,
                    data={},
                    confidence=0.0,
                    timestamp=datetime.now(),
                    notes=response.get('error', 'LLM response failed')
                )
            
            # Parse and structure the response
            detailed_data = self._parse_detailed_analysis(response['recommendation'])
            
            return AnalysisResult(
                step=AnalysisStep.DETAILED_ANALYSIS,
                success=True,
                data=detailed_data,
                confidence=detailed_data.get('confidence', 0.8),
                timestamp=datetime.now(),
                notes="Detailed analysis completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in detailed analysis: {str(e)}")
            return AnalysisResult(
                step=AnalysisStep.DETAILED_ANALYSIS,
                success=False,
                data={},
                confidence=0.0,
                timestamp=datetime.now(),
                notes=str(e)
            )
    
    def _run_treatment_planning(self, 
                               skin_analysis: Dict[str, Any],
                               user_profile: Optional[Dict[str, Any]],
                               detailed_data: Dict[str, Any]) -> AnalysisResult:
        """
        Create comprehensive treatment plan
        
        Args:
            skin_analysis: Skin analysis results
            user_profile: User profile information
            detailed_data: Data from detailed analysis
            
        Returns:
            Treatment planning result
        """
        try:
            # Create treatment planning prompt
            prompt = self._create_treatment_planning_prompt(skin_analysis, user_profile, detailed_data)
            
            # Get LLM response
            response = self.llm_connector.generate_recommendation(
                skin_analysis, user_profile, prompt
            )
            
            if not response.get('success'):
                return AnalysisResult(
                    step=AnalysisStep.TREATMENT_PLANNING,
                    success=False,
                    data={},
                    confidence=0.0,
                    timestamp=datetime.now(),
                    notes=response.get('error', 'LLM response failed')
                )
            
            # Parse and structure the response
            treatment_data = self._parse_treatment_planning(response['recommendation'])
            
            return AnalysisResult(
                step=AnalysisStep.TREATMENT_PLANNING,
                success=True,
                data=treatment_data,
                confidence=treatment_data.get('confidence', 0.85),
                timestamp=datetime.now(),
                notes="Treatment planning completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in treatment planning: {str(e)}")
            return AnalysisResult(
                step=AnalysisStep.TREATMENT_PLANNING,
                success=False,
                data={},
                confidence=0.0,
                timestamp=datetime.now(),
                notes=str(e)
            )
    
    def _run_product_selection(self, 
                              treatment_data: Dict[str, Any],
                              user_profile: Optional[Dict[str, Any]]) -> AnalysisResult:
        """
        Select specific products based on treatment plan
        
        Args:
            treatment_data: Treatment planning data
            user_profile: User profile information
            
        Returns:
            Product selection result
        """
        try:
            # Create product selection prompt
            prompt = self._create_product_selection_prompt(treatment_data, user_profile)
            
            # Get LLM response
            response = self.llm_connector.generate_recommendation(
                {'treatment_plan': treatment_data}, user_profile, prompt
            )
            
            if not response.get('success'):
                return AnalysisResult(
                    step=AnalysisStep.PRODUCT_SELECTION,
                    success=False,
                    data={},
                    confidence=0.0,
                    timestamp=datetime.now(),
                    notes=response.get('error', 'LLM response failed')
                )
            
            # Parse and structure the response
            product_data = self._parse_product_selection(response['recommendation'])
            
            return AnalysisResult(
                step=AnalysisStep.PRODUCT_SELECTION,
                success=True,
                data=product_data,
                confidence=product_data.get('confidence', 0.8),
                timestamp=datetime.now(),
                notes="Product selection completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in product selection: {str(e)}")
            return AnalysisResult(
                step=AnalysisStep.PRODUCT_SELECTION,
                success=False,
                data={},
                confidence=0.0,
                timestamp=datetime.now(),
                notes=str(e)
            )
    
    def _run_lifestyle_integration(self, 
                                  treatment_data: Dict[str, Any],
                                  user_profile: Optional[Dict[str, Any]]) -> AnalysisResult:
        """
        Integrate lifestyle recommendations with treatment plan
        
        Args:
            treatment_data: Treatment planning data
            user_profile: User profile information
            
        Returns:
            Lifestyle integration result
        """
        try:
            # Create lifestyle integration prompt
            prompt = self._create_lifestyle_integration_prompt(treatment_data, user_profile)
            
            # Get LLM response
            response = self.llm_connector.generate_recommendation(
                {'treatment_plan': treatment_data}, user_profile, prompt
            )
            
            if not response.get('success'):
                return AnalysisResult(
                    step=AnalysisStep.LIFESTYLE_INTEGRATION,
                    success=False,
                    data={},
                    confidence=0.0,
                    timestamp=datetime.now(),
                    notes=response.get('error', 'LLM response failed')
                )
            
            # Parse and structure the response
            lifestyle_data = self._parse_lifestyle_integration(response['recommendation'])
            
            return AnalysisResult(
                step=AnalysisStep.LIFESTYLE_INTEGRATION,
                success=True,
                data=lifestyle_data,
                confidence=lifestyle_data.get('confidence', 0.75),
                timestamp=datetime.now(),
                notes="Lifestyle integration completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in lifestyle integration: {str(e)}")
            return AnalysisResult(
                step=AnalysisStep.LIFESTYLE_INTEGRATION,
                success=False,
                data={},
                confidence=0.0,
                timestamp=datetime.now(),
                notes=str(e)
            )
    
    def _run_follow_up_planning(self, 
                               treatment_data: Dict[str, Any],
                               user_profile: Optional[Dict[str, Any]]) -> AnalysisResult:
        """
        Create follow-up and monitoring plan
        
        Args:
            treatment_data: Treatment planning data
            user_profile: User profile information
            
        Returns:
            Follow-up planning result
        """
        try:
            # Create follow-up planning prompt
            prompt = self._create_follow_up_planning_prompt(treatment_data, user_profile)
            
            # Get LLM response
            response = self.llm_connector.generate_recommendation(
                {'treatment_plan': treatment_data}, user_profile, prompt
            )
            
            if not response.get('success'):
                return AnalysisResult(
                    step=AnalysisStep.FOLLOW_UP_PLANNING,
                    success=False,
                    data={},
                    confidence=0.0,
                    timestamp=datetime.now(),
                    notes=response.get('error', 'LLM response failed')
                )
            
            # Parse and structure the response
            follow_up_data = self._parse_follow_up_planning(response['recommendation'])
            
            return AnalysisResult(
                step=AnalysisStep.FOLLOW_UP_PLANNING,
                success=True,
                data=follow_up_data,
                confidence=follow_up_data.get('confidence', 0.8),
                timestamp=datetime.now(),
                notes="Follow-up planning completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error in follow-up planning: {str(e)}")
            return AnalysisResult(
                step=AnalysisStep.FOLLOW_UP_PLANNING,
                success=False,
                data={},
                confidence=0.0,
                timestamp=datetime.now(),
                notes=str(e)
            )
    
    def _create_final_recommendation(self, 
                                   skin_analysis: Dict[str, Any],
                                   user_profile: Optional[Dict[str, Any]],
                                   session_id: str) -> TreatmentPlan:
        """
        Create final comprehensive treatment plan
        
        Args:
            skin_analysis: Skin analysis results
            user_profile: User profile information
            session_id: Session identifier
            
        Returns:
            Complete treatment plan
        """
        # Extract data from analysis steps
        treatment_data = {}
        product_data = {}
        lifestyle_data = {}
        follow_up_data = {}
        
        for step in self.analysis_history:
            if step.step == AnalysisStep.TREATMENT_PLANNING:
                treatment_data = step.data
            elif step.step == AnalysisStep.PRODUCT_SELECTION:
                product_data = step.data
            elif step.step == AnalysisStep.LIFESTYLE_INTEGRATION:
                lifestyle_data = step.data
            elif step.step == AnalysisStep.FOLLOW_UP_PLANNING:
                follow_up_data = step.data
        
        # Create comprehensive treatment plan
        # Normalize product recommendations structure
        norm_products = {}
        if isinstance(product_data, dict):
            if 'products' in product_data and isinstance(product_data['products'], dict):
                norm_products = product_data['products']
            elif 'product_recommendations' in product_data:
                pr = product_data['product_recommendations']
                if isinstance(pr, list):
                    norm_products = { 'general': pr }
                elif isinstance(pr, dict):
                    norm_products = pr
        
        plan = TreatmentPlan(
            plan_id=f"plan_{session_id}",
            user_id=user_profile.get('user_id', 'unknown') if user_profile else 'unknown',
            created_at=datetime.now(),
            skin_analysis=skin_analysis,
            user_profile=user_profile or {},
            treatment_steps=treatment_data.get('treatment_steps', []),
            product_recommendations=norm_products,
            lifestyle_recommendations=lifestyle_data.get('lifestyle_tips', []),
            follow_up_schedule=follow_up_data.get('follow_up_schedule', {}),
            priority_actions=treatment_data.get('priority_actions', []),
            risk_factors=treatment_data.get('risk_factors', []),
            expected_timeline=treatment_data.get('expected_timeline', '4-6 weeks'),
            success_metrics=treatment_data.get('success_metrics', {})
        )
        
        self.current_plan = plan
        return plan
    
    def _create_initial_assessment_prompt(self, 
                                        skin_analysis: Dict[str, Any],
                                        user_profile: Optional[Dict[str, Any]]) -> str:
        """Create prompt for initial assessment"""
        return f"""You are an expert dermatologist conducting an initial assessment. Analyze the following skin condition and provide a structured assessment.

Skin Analysis: {json.dumps(skin_analysis, indent=2)}
User Profile: {json.dumps(user_profile, indent=2) if user_profile else 'Not provided'}

Please provide your assessment in the following JSON format:
{{
    "overall_severity": "mild/moderate/severe/critical",
    "primary_concerns": ["list of main issues"],
    "secondary_concerns": ["list of secondary issues"],
    "urgency_level": "low/medium/high/critical",
    "requires_immediate_attention": true/false,
    "confidence": 0.0-1.0,
    "notes": "Brief assessment notes"
}}"""
    
    def _create_detailed_analysis_prompt(self, 
                                       skin_analysis: Dict[str, Any],
                                       user_profile: Optional[Dict[str, Any]],
                                       initial_data: Dict[str, Any]) -> str:
        """Create prompt for detailed analysis"""
        return f"""You are an expert dermatologist conducting a detailed analysis. Based on the initial assessment, provide a comprehensive analysis.

Initial Assessment: {json.dumps(initial_data, indent=2)}
Skin Analysis: {json.dumps(skin_analysis, indent=2)}
User Profile: {json.dumps(user_profile, indent=2) if user_profile else 'Not provided'}

Please provide detailed analysis in the following JSON format:
{{
    "root_causes": ["underlying causes of skin issues"],
    "contributing_factors": ["factors that may worsen the condition"],
    "skin_type_assessment": "oily/dry/combination/sensitive/normal",
    "barrier_function": "intact/compromised/severely_damaged",
    "inflammation_level": "none/mild/moderate/severe",
    "treatment_complexity": "simple/moderate/complex",
    "confidence": 0.0-1.0,
    "analysis_notes": "Detailed analysis notes"
}}"""
    
    def _create_treatment_planning_prompt(self, 
                                        skin_analysis: Dict[str, Any],
                                        user_profile: Optional[Dict[str, Any]],
                                        detailed_data: Dict[str, Any]) -> str:
        """Create prompt for treatment planning"""
        return f"""You are an expert dermatologist creating a comprehensive treatment plan. Based on the detailed analysis, develop a personalized treatment strategy.

Detailed Analysis: {json.dumps(detailed_data, indent=2)}
Skin Analysis: {json.dumps(skin_analysis, indent=2)}
User Profile: {json.dumps(user_profile, indent=2) if user_profile else 'Not provided'}

Please provide treatment plan in the following JSON format:
{{
    "treatment_phases": [
        {{
            "phase": "phase_name",
            "duration": "duration_in_weeks",
            "goals": ["specific goals"],
            "approach": "treatment approach description"
        }}
    ],
    "priority_actions": ["immediate actions to take"],
    "risk_factors": ["potential risks or complications"],
    "expected_timeline": "expected timeline for improvement",
    "success_metrics": {{
        "primary_goals": ["main improvement goals"],
        "secondary_goals": ["additional improvement goals"],
        "warning_signs": ["signs to watch for"]
    }},
    "confidence": 0.0-1.0,
    "treatment_notes": "Treatment planning notes"
}}"""
    
    def _create_product_selection_prompt(self, 
                                       treatment_data: Dict[str, Any],
                                       user_profile: Optional[Dict[str, Any]]) -> str:
        """Create prompt for product selection"""
        return f"""You are an expert dermatologist selecting specific products for the treatment plan. Recommend specific products and ingredients.

Treatment Plan: {json.dumps(treatment_data, indent=2)}
User Profile: {json.dumps(user_profile, indent=2) if user_profile else 'Not provided'}

Please provide product recommendations in the following JSON format:
{{
    "products": {{
        "cleansers": ["specific product recommendations"],
        "treatments": ["active ingredients and treatments"],
        "moisturizers": ["moisturizer recommendations"],
        "sunscreens": ["sunscreen recommendations"],
        "masks": ["mask recommendations"],
        "supplements": ["supplement recommendations"]
    }},
    "ingredient_priorities": ["most important ingredients"],
    "budget_considerations": "budget-friendly alternatives",
    "confidence": 0.0-1.0,
    "product_notes": "Product selection notes"
}}"""
    
    def _create_lifestyle_integration_prompt(self, 
                                           treatment_data: Dict[str, Any],
                                           user_profile: Optional[Dict[str, Any]]) -> str:
        """Create prompt for lifestyle integration"""
        return f"""You are an expert dermatologist integrating lifestyle recommendations with the treatment plan. Provide lifestyle modifications to support skin health.

Treatment Plan: {json.dumps(treatment_data, indent=2)}
User Profile: {json.dumps(user_profile, indent=2) if user_profile else 'Not provided'}

Please provide lifestyle recommendations in the following JSON format:
{{
    "lifestyle_tips": ["specific lifestyle recommendations"],
    "diet_recommendations": ["dietary changes"],
    "stress_management": ["stress reduction techniques"],
    "sleep_optimization": ["sleep improvement tips"],
    "environmental_protection": ["environmental factors to consider"],
    "exercise_recommendations": ["exercise-related advice"],
    "confidence": 0.0-1.0,
    "lifestyle_notes": "Lifestyle integration notes"
}}"""
    
    def _create_follow_up_planning_prompt(self, 
                                        treatment_data: Dict[str, Any],
                                        user_profile: Optional[Dict[str, Any]]) -> str:
        """Create prompt for follow-up planning"""
        return f"""You are an expert dermatologist creating a follow-up and monitoring plan. Develop a schedule for tracking progress and adjusting treatment.

Treatment Plan: {json.dumps(treatment_data, indent=2)}
User Profile: {json.dumps(user_profile, indent=2) if user_profile else 'Not provided'}

Please provide follow-up plan in the following JSON format:
{{
    "follow_up_schedule": {{
        "weekly_check_ins": ["weekly monitoring tasks"],
        "monthly_assessments": ["monthly evaluation points"],
        "progress_indicators": ["signs of improvement"],
        "adjustment_triggers": ["when to modify treatment"]
    }},
    "monitoring_tools": ["tools for tracking progress"],
    "emergency_signs": ["signs requiring immediate attention"],
    "long_term_plan": ["long-term maintenance strategy"],
    "confidence": 0.0-1.0,
    "follow_up_notes": "Follow-up planning notes"
}}"""
    
    def _parse_initial_assessment(self, response: str) -> Dict[str, Any]:
        """Parse initial assessment response"""
        try:
            if isinstance(response, str):
                return json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Error parsing initial assessment: {str(e)}")
            return {"confidence": 0.5, "notes": "Parsing failed"}
    
    def _parse_detailed_analysis(self, response: str) -> Dict[str, Any]:
        """Parse detailed analysis response"""
        try:
            if isinstance(response, str):
                return json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Error parsing detailed analysis: {str(e)}")
            return {"confidence": 0.5, "notes": "Parsing failed"}
    
    def _parse_treatment_planning(self, response: str) -> Dict[str, Any]:
        """Parse treatment planning response"""
        try:
            if isinstance(response, str):
                return json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Error parsing treatment planning: {str(e)}")
            return {"confidence": 0.5, "notes": "Parsing failed"}
    
    def _parse_product_selection(self, response: str) -> Dict[str, Any]:
        """Parse product selection response"""
        try:
            # Convert string JSON to dict if needed
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except Exception:
                    # If it's plain text, wrap as a generic list under general
                    return {
                        "products": {"general": [response]}
                    }

            # If LLM/demo returned a flat list under 'product_recommendations'
            if isinstance(response, dict):
                # Case 1: Already categorized under 'products'
                if 'products' in response and isinstance(response['products'], dict):
                    return response

                # Case 2: List under 'product_recommendations'
                if 'product_recommendations' in response and isinstance(response['product_recommendations'], list):
                    return {
                        'products': {
                            'general': response['product_recommendations']
                        },
                        'confidence': response.get('confidence', 0.8)
                    }

                # Case 3: Single string under 'product_recommendations'
                if 'product_recommendations' in response and isinstance(response['product_recommendations'], str):
                    return {
                        'products': {
                            'general': [response['product_recommendations']]
                        },
                        'confidence': response.get('confidence', 0.8)
                    }

            # Fallback: if it's a list, map to general
            if isinstance(response, list):
                return { 'products': { 'general': response } }

            # Last resort: empty structure
            return { 'products': {} }
        except Exception as e:
            logger.error(f"Error parsing product selection: {str(e)}")
            return {"confidence": 0.5, "notes": "Parsing failed"}
    
    def _parse_lifestyle_integration(self, response: str) -> Dict[str, Any]:
        """Parse lifestyle integration response"""
        try:
            if isinstance(response, str):
                return json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Error parsing lifestyle integration: {str(e)}")
            return {"confidence": 0.5, "notes": "Parsing failed"}
    
    def _parse_follow_up_planning(self, response: str) -> Dict[str, Any]:
        """Parse follow-up planning response"""
        try:
            if isinstance(response, str):
                return json.loads(response)
            return response
        except Exception as e:
            logger.error(f"Error parsing follow-up planning: {str(e)}")
            return {"confidence": 0.5, "notes": "Parsing failed"}
    
    def _create_error_response(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'analysis_steps': [asdict(step) for step in self.analysis_history]
        }
    
    def _create_analysis_summary(self) -> Dict[str, Any]:
        """Create summary of the analysis process"""
        successful_steps = [step for step in self.analysis_history if step.success]
        failed_steps = [step for step in self.analysis_history if not step.success]
        
        return {
            'total_steps': len(self.analysis_history),
            'successful_steps': len(successful_steps),
            'failed_steps': len(failed_steps),
            'success_rate': len(successful_steps) / len(self.analysis_history) if self.analysis_history else 0,
            'average_confidence': sum(step.confidence for step in successful_steps) / len(successful_steps) if successful_steps else 0,
            'completion_time': (self.analysis_history[-1].timestamp - self.analysis_history[0].timestamp).total_seconds() if len(self.analysis_history) > 1 else 0
        }
    
    def _get_next_actions(self, treatment_plan: TreatmentPlan) -> List[str]:
        """Get immediate next actions for the user"""
        actions = []
        
        # Add priority actions
        actions.extend(treatment_plan.priority_actions[:3])
        
        # Add immediate product recommendations
        if treatment_plan.product_recommendations.get('cleansers'):
            actions.append(f"Purchase recommended cleanser: {treatment_plan.product_recommendations['cleansers'][0]}")
        
        # Add lifestyle recommendations
        if treatment_plan.lifestyle_recommendations:
            actions.append(f"Start implementing: {treatment_plan.lifestyle_recommendations[0]}")
        
        return actions[:5]  # Limit to top 5 actions
    
    def _assess_risks(self, treatment_plan: TreatmentPlan) -> Dict[str, Any]:
        """Assess potential risks and complications"""
        risk_level = "low"
        if any("severe" in risk.lower() for risk in treatment_plan.risk_factors):
            risk_level = "high"
        elif any("moderate" in risk.lower() for risk in treatment_plan.risk_factors):
            risk_level = "medium"
        
        return {
            'overall_risk_level': risk_level,
            'risk_factors': treatment_plan.risk_factors,
            'mitigation_strategies': [
                "Follow treatment plan exactly as prescribed",
                "Monitor for any adverse reactions",
                "Consult dermatologist if concerns arise"
            ],
            'emergency_contacts': "Seek immediate medical attention for severe reactions"
        }
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score"""
        successful_steps = [step for step in self.analysis_history if step.success]
        if not successful_steps:
            return 0.0
        
        return sum(step.confidence for step in successful_steps) / len(successful_steps)

    def _generate_rule_based_recommendations(self, 
                                           skin_analysis: Dict[str, Any],
                                           user_profile: Optional[Dict[str, Any]] = None,
                                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate recommendations using rule-based system (no LLM required)
        """
        try:
            session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Extract detected issues
            detected_issues = skin_analysis.get('detected_issues', [])
            primary_prediction = skin_analysis.get('primary_prediction', {})
            primary_issue = primary_prediction.get('class', 'normal_skin')
            
            # Generate treatment plan based on detected issues
            treatment_steps = self._generate_treatment_steps(detected_issues)
            product_recommendations = self._generate_product_recommendations(detected_issues)
            lifestyle_recommendations = self._generate_lifestyle_recommendations(detected_issues)
            priority_actions = self._generate_priority_actions(detected_issues)
            risk_factors = self._generate_risk_factors(detected_issues)
            
            # Create treatment plan
            treatment_plan = TreatmentPlan(
                plan_id=f"plan_{session_id}",
                user_id="user_001",
                created_at=datetime.now(),
                skin_analysis=skin_analysis,
                user_profile=user_profile or {},
                treatment_steps=treatment_steps,
                product_recommendations=product_recommendations,
                lifestyle_recommendations=lifestyle_recommendations,
                follow_up_schedule=self._generate_follow_up_schedule(detected_issues),
                priority_actions=priority_actions,
                risk_factors=risk_factors,
                expected_timeline=self._generate_timeline(detected_issues),
                success_metrics=self._generate_success_metrics(detected_issues)
            )
            
            return {
                'success': True,
                'detected_problems': self._format_detected_problems(detected_issues),
                'treatment_plan': self._generate_detailed_treatment_plan(detected_issues),
                'product_recommendations': product_recommendations,
                'lifestyle_recommendations': lifestyle_recommendations,
                'priority_actions': priority_actions,
                'timeline': self._generate_timeline(detected_issues),
                'follow_up_schedule': self._generate_follow_up_schedule(detected_issues),
                'analysis_summary': {
                    'total_issues': len(detected_issues),
                    'primary_concern': primary_issue,
                    'severity_level': self._assess_severity(detected_issues),
                    'confidence_score': primary_prediction.get('confidence', 0.5)
                },
                'confidence_score': primary_prediction.get('confidence', 0.5),
                'risk_assessment': self._assess_risks(treatment_plan),
                'next_actions': self._get_next_actions(treatment_plan),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based recommendations: {str(e)}")
            return self._create_error_response("Rule-based analysis failed", str(e))
    
    def _generate_treatment_steps(self, detected_issues: List[str]) -> List[Dict[str, Any]]:
        """Generate treatment steps based on detected issues"""
        steps = []
        
        if 'acne' in detected_issues or 'pimples' in detected_issues:
            steps.append({
                'phase': 'Acne Treatment',
                'duration': '4-8 weeks',
                'goals': ['Reduce inflammation', 'Prevent new breakouts', 'Heal existing lesions'],
                'approach': 'Use salicylic acid cleanser and benzoyl peroxide spot treatment'
            })
        
        if 'dark_spots' in detected_issues or 'hyperpigmentation' in detected_issues:
            steps.append({
                'phase': 'Brightening Treatment',
                'duration': '8-12 weeks',
                'goals': ['Fade dark spots', 'Even skin tone', 'Prevent new spots'],
                'approach': 'Use vitamin C serum and sunscreen daily'
            })
        
        if 'dry_skin' in detected_issues:
            steps.append({
                'phase': 'Hydration Treatment',
                'duration': '2-4 weeks',
                'goals': ['Restore moisture', 'Strengthen barrier', 'Prevent flaking'],
                'approach': 'Use gentle cleanser and rich moisturizer'
            })
        
        if 'oily_skin' in detected_issues:
            steps.append({
                'phase': 'Oil Control',
                'duration': '4-6 weeks',
                'goals': ['Reduce excess oil', 'Prevent clogged pores', 'Maintain balance'],
                'approach': 'Use oil-free products and clay masks'
            })
        
        if not steps:
            steps.append({
                'phase': 'Maintenance',
                'duration': 'Ongoing',
                'goals': ['Maintain healthy skin', 'Prevent issues', 'Protect from damage'],
                'approach': 'Use gentle cleanser, moisturizer, and sunscreen'
            })
        
        return steps
    
    def _generate_product_recommendations(self, detected_issues: List[str]) -> Dict[str, List[str]]:
        """Generate product recommendations based on detected issues"""
        recommendations = {
            'cleansers': [],
            'treatments': [],
            'moisturizers': [],
            'sunscreens': [],
            'masks': []
        }
        
        # Cleansers
        if 'acne' in detected_issues or 'oily_skin' in detected_issues:
            recommendations['cleansers'].extend([
                'Salicylic acid cleanser',
                'Oil-free foaming cleanser'
            ])
        elif 'dry_skin' in detected_issues:
            recommendations['cleansers'].extend([
                'Gentle cream cleanser',
                'Hydrating cleanser'
            ])
        else:
            recommendations['cleansers'].append('Gentle daily cleanser')
        
        # Treatments
        if 'acne' in detected_issues:
            recommendations['treatments'].extend([
                'Benzoyl peroxide spot treatment',
                'Salicylic acid toner'
            ])
        
        if 'dark_spots' in detected_issues:
            recommendations['treatments'].extend([
                'Vitamin C serum',
                'Niacinamide serum'
            ])
        
        if 'dry_skin' in detected_issues:
            recommendations['treatments'].append('Hyaluronic acid serum')
        
        # Moisturizers
        if 'dry_skin' in detected_issues:
            recommendations['moisturizers'].append('Rich hydrating moisturizer')
        elif 'oily_skin' in detected_issues:
            recommendations['moisturizers'].append('Oil-free gel moisturizer')
        else:
            recommendations['moisturizers'].append('Lightweight moisturizer')
        
        # Sunscreens
        recommendations['sunscreens'].append('Broad-spectrum SPF 30+ sunscreen')
        
        # Masks
        if 'oily_skin' in detected_issues:
            recommendations['masks'].append('Clay mask (1-2 times per week)')
        elif 'dry_skin' in detected_issues:
            recommendations['masks'].append('Hydrating sheet mask (1-2 times per week)')
        
        return recommendations
    
    def _generate_lifestyle_recommendations(self, detected_issues: List[str]) -> List[str]:
        """Generate lifestyle recommendations"""
        recommendations = [
            'Drink 8 glasses of water daily',
            'Get 7-9 hours of sleep',
            'Manage stress through meditation or exercise',
            'Avoid touching your face frequently',
            'Change pillowcases weekly'
        ]
        
        if 'acne' in detected_issues:
            recommendations.extend([
                'Avoid dairy if it triggers breakouts',
                'Use non-comedogenic makeup',
                'Clean makeup brushes regularly'
            ])
        
        if 'dark_spots' in detected_issues:
            recommendations.extend([
                'Wear wide-brimmed hat outdoors',
                'Avoid sun exposure during peak hours',
                'Use antioxidant-rich diet'
            ])
        
        return recommendations
    
    def _generate_priority_actions(self, detected_issues: List[str]) -> List[str]:
        """Generate priority actions"""
        actions = [
            'Start using recommended products',
            'Establish consistent skincare routine',
            'Monitor skin changes weekly'
        ]
        
        if 'acne' in detected_issues:
            actions.insert(0, 'Begin acne treatment immediately')
        
        if 'dark_spots' in detected_issues:
            actions.insert(0, 'Start brightening treatment')
        
        return actions
    
    def _generate_risk_factors(self, detected_issues: List[str]) -> List[str]:
        """Generate risk factors"""
        risks = ['Sun damage', 'Environmental pollution']
        
        if 'acne' in detected_issues:
            risks.extend(['Scarring from picking', 'Post-inflammatory hyperpigmentation'])
        
        if 'dark_spots' in detected_issues:
            risks.extend(['Sun exposure worsening spots', 'Inflammation from harsh products'])
        
        return risks
    
    def _generate_follow_up_schedule(self, detected_issues: List[str]) -> Dict[str, Any]:
        """Generate follow-up schedule"""
        return {
            'weekly_check': 'Monitor progress and adjust routine',
            'monthly_review': 'Assess treatment effectiveness',
            'three_month_evaluation': 'Consider professional consultation if needed'
        }
    
    def _generate_timeline(self, detected_issues: List[str]) -> str:
        """Generate expected timeline"""
        if 'acne' in detected_issues:
            return '4-8 weeks for significant improvement'
        elif 'dark_spots' in detected_issues:
            return '8-12 weeks for visible fading'
        elif 'dry_skin' in detected_issues:
            return '2-4 weeks for hydration improvement'
        else:
            return '2-4 weeks for overall improvement'
    
    def _generate_success_metrics(self, detected_issues: List[str]) -> Dict[str, Any]:
        """Generate success metrics"""
        return {
            'primary_goals': [
                'Reduced appearance of detected issues',
                'Improved skin texture and tone',
                'Maintained skin health'
            ],
            'warning_signs': [
                'Increased irritation or redness',
                'No improvement after 4 weeks',
                'Worsening of existing conditions'
            ]
        }
    
    def _assess_severity(self, detected_issues: List[str]) -> str:
        """Assess severity level"""
        if len(detected_issues) >= 3:
            return 'moderate'
        elif len(detected_issues) >= 1:
            return 'mild'
        else:
            return 'minimal'
    
    def _format_detected_problems(self, detected_issues: List[str]) -> Dict[str, Any]:
        """Format detected problems with descriptions and severity"""
        problem_descriptions = {
            'acne': {
                'name': 'Acne/Pimples',
                'description': 'Inflammatory skin condition with pimples, blackheads, and whiteheads',
                'severity': 'Medium',
                'common_causes': ['Excess oil production', 'Bacteria', 'Hormonal changes', 'Poor hygiene']
            },
            'dark_spots': {
                'name': 'Dark Spots/Hyperpigmentation',
                'description': 'Darkened areas of skin due to excess melanin production',
                'severity': 'Medium',
                'common_causes': ['Sun exposure', 'Acne scars', 'Hormonal changes', 'Skin trauma']
            },
            'dry_skin': {
                'name': 'Dry Skin',
                'description': 'Skin lacking sufficient moisture and natural oils',
                'severity': 'Low',
                'common_causes': ['Harsh weather', 'Hot showers', 'Harsh soaps', 'Dehydration']
            },
            'oily_skin': {
                'name': 'Oily Skin',
                'description': 'Excess sebum production making skin appear shiny',
                'severity': 'Medium',
                'common_causes': ['Genetics', 'Hormonal changes', 'Humidity', 'Harsh products']
            },
            'normal_skin': {
                'name': 'Normal Skin',
                'description': 'Well-balanced skin with good moisture and oil levels',
                'severity': 'None',
                'common_causes': ['Good genetics', 'Proper care', 'Healthy lifestyle']
            }
        }
        
        formatted_problems = []
        for issue in detected_issues:
            if issue in problem_descriptions:
                formatted_problems.append(problem_descriptions[issue])
            else:
                formatted_problems.append({
                    'name': issue.replace('_', ' ').title(),
                    'description': f'Detected skin issue: {issue}',
                    'severity': 'Medium',
                    'common_causes': ['Various factors']
                })
        
        return {
            'total_problems': len(formatted_problems),
            'problems': formatted_problems,
            'primary_concern': formatted_problems[0]['name'] if formatted_problems else 'None detected'
        }
    
    def _generate_detailed_treatment_plan(self, detected_issues: List[str]) -> str:
        """Generate detailed treatment plan text"""
        plan_parts = []
        
        if 'acne' in detected_issues:
            plan_parts.append("""
**Acne Treatment Plan:**
- **Morning Routine:** Gentle cleanser → Salicylic acid toner → Light moisturizer → Sunscreen
- **Evening Routine:** Double cleanse → Benzoyl peroxide spot treatment → Oil-free moisturizer
- **Weekly:** Clay mask to absorb excess oil
- **Avoid:** Touching face, picking pimples, heavy makeup
            """)
        
        if 'dark_spots' in detected_issues:
            plan_parts.append("""
**Dark Spots Treatment Plan:**
- **Morning Routine:** Gentle cleanser → Vitamin C serum → Niacinamide → Moisturizer → SPF 30+
- **Evening Routine:** Double cleanse → Alpha hydroxy acid (AHA) → Brightening serum → Moisturizer
- **Weekly:** Exfoliating treatment (1-2 times)
- **Key:** Consistent sun protection is crucial
            """)
        
        if 'dry_skin' in detected_issues:
            plan_parts.append("""
**Dry Skin Treatment Plan:**
- **Morning Routine:** Gentle cream cleanser → Hyaluronic acid serum → Rich moisturizer → Sunscreen
- **Evening Routine:** Oil-based cleanser → Hydrating toner → Face oil → Thick moisturizer
- **Weekly:** Hydrating sheet mask (2-3 times)
- **Avoid:** Hot water, harsh soaps, alcohol-based products
            """)
        
        if 'oily_skin' in detected_issues:
            plan_parts.append("""
**Oily Skin Treatment Plan:**
- **Morning Routine:** Foaming cleanser → BHA toner → Oil-free moisturizer → Matte sunscreen
- **Evening Routine:** Double cleanse → Clay-based treatment → Light gel moisturizer
- **Weekly:** Clay mask to control oil
- **Avoid:** Heavy creams, over-cleansing, harsh scrubs
            """)
        
        if not plan_parts:
            plan_parts.append("""
**General Skin Care Plan:**
- **Morning Routine:** Gentle cleanser → Moisturizer → Sunscreen
- **Evening Routine:** Cleanse → Moisturizer
- **Weekly:** Gentle exfoliation (1 time)
- **Focus:** Maintaining healthy skin barrier
            """)
        
        return "\n".join(plan_parts)


# Utility functions
def create_agentic_flow(llm_api_key: Optional[str] = None) -> AgenticDermatologistFlow:
    """
    Create an agentic dermatologist flow instance
    
    Args:
        llm_api_key: OpenAI API key
        
    Returns:
        AgenticDermatologistFlow instance
    """
    llm_connector = LLMConnector(api_key=llm_api_key)
    return AgenticDermatologistFlow(llm_connector)

def run_quick_analysis(skin_analysis: Dict[str, Any],
                      user_profile: Optional[Dict[str, Any]] = None,
                      llm_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a quick analysis using the agentic flow
    
    Args:
        skin_analysis: Skin analysis results
        user_profile: User profile information
        llm_api_key: OpenAI API key
        
    Returns:
        Quick analysis results
    """
    flow = create_agentic_flow(llm_api_key)
    return flow.run_complete_analysis(skin_analysis, user_profile)


# Example usage
if __name__ == "__main__":
    # Test the agentic flow
    sample_analysis = {
        'detected_issues': [
            {
                'issue': 'acne',
                'display_name': 'Acne',
                'confidence': 0.85,
                'severity': 'moderate'
            }
        ],
        'overall_condition': {
            'summary': 'Moderate acne detected'
        },
        'category_analysis': {
            'acne_related': {
                'count': 1,
                'total_confidence': 0.85
            }
        }
    }
    
    sample_profile = {
        'age': 25,
        'skin_type': 'combination',
        'concerns': ['acne', 'oiliness'],
        'budget': 'medium'
    }
    
    # Run quick analysis
    result = run_quick_analysis(sample_analysis, sample_profile)
    print("Agentic Analysis Result:")
    print(json.dumps(result, indent=2)) 