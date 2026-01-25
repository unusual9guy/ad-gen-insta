"""Agent 1: Product Analyzer - Uses Gemini to analyze product images."""

import json
import base64
from typing import Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from agents.base import BaseAgent
from workflow.state import WorkflowState, ProductAnalysis
from config.settings import settings


PRODUCT_ANALYSIS_PROMPT = """You are an expert product photographer and visual analyst. Analyze this product image in detail and provide a comprehensive analysis for creating a professional advertisement.

Product Name: {product_name}

Please analyze the image and return a JSON object with the following structure:

{{
    "product_type": "The specific type of product (e.g., 'wireless headphones', 'skincare serum', 'running shoes')",
    "product_category": "Broad category (e.g., 'electronics', 'beauty', 'footwear')",
    "current_angle": "Description of the current viewing angle (e.g., 'front-facing', '3/4 view', 'top-down')",
    "visual_characteristics": {{
        "shape": "Overall shape description",
        "texture": "Surface texture description",
        "finish": "Surface finish (matte, glossy, metallic, etc.)",
        "size_impression": "Perceived size (compact, medium, large)"
    }},
    "colors": ["List of primary colors in the product"],
    "materials": ["List of apparent materials (plastic, metal, fabric, etc.)"],
    "positioning_recommendation": "Recommended positioning/angle for the final ad composition",
    "composition_notes": "Notes on how to best showcase this product in an ad (lighting suggestions, background compatibility, focal points)",
    "needs_angle_regeneration": false,
    "angle_regeneration_reason": null
}}

Important guidelines:
1. Be specific and detailed in your analysis
2. Set needs_angle_regeneration to true ONLY if the current image angle is truly unsuitable for advertising (e.g., blurry, poorly lit, unflattering angle)
3. Provide actionable positioning recommendations that will help create an effective ad
4. Consider the product's best features and how to highlight them

Return ONLY the JSON object, no additional text or markdown formatting."""


class ProductAnalyzerAgent(BaseAgent[ProductAnalysis]):
    """
    Agent 1: Analyzes product images using Gemini Vision.
    
    This agent extracts detailed information about the product including:
    - Product type and category
    - Visual characteristics
    - Color palette
    - Materials
    - Positioning recommendations for the final ad
    """
    
    name = "ProductAnalyzer"
    description = "Analyzes product images to extract visual characteristics and positioning recommendations"
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model for Agent 1."""
        api_key = settings.get_agent1_key()
        if api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-image",
                google_api_key=api_key,
                temperature=0.3,
                max_tokens=2048
            )
            self.logger.info("Agent 1 (Product Analyzer) initialized with Gemini")
        else:
            self.logger.warning("No API key configured for Agent 1 (Product Analyzer)")
    
    def validate_inputs(self, state: WorkflowState) -> bool:
        """Validate that product image and name are present."""
        if not state.get("product_image"):
            self.logger.error("Product image is required")
            return False
        if not state.get("product_name"):
            self.logger.error("Product name is required")
            return False
        return True
    
    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """
        Analyze the product image and return structured analysis.
        
        Args:
            state: Current workflow state with product_image and product_name
            
        Returns:
            Dictionary containing product_analysis
        """
        if not self.llm:
            raise RuntimeError("Gemini model not initialized. Check API key configuration.")
        
        # Prepare the image for Gemini
        image_bytes = state["product_image"]
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create the prompt with product name
        prompt = PRODUCT_ANALYSIS_PROMPT.format(product_name=state["product_name"])
        
        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        )
        
        # Generate analysis
        response = self.llm.invoke([message])
        
        # Parse the JSON response
        response_text = response.content.strip()
        
        # Clean up response if wrapped in markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (```json and ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        try:
            analysis_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Gemini response as JSON: {e}")
            self.logger.debug(f"Response text: {response_text}")
            raise ValueError(f"Failed to parse product analysis: {e}")
        
        # Validate and structure the analysis
        product_analysis: ProductAnalysis = {
            "product_type": analysis_data.get("product_type", "unknown"),
            "product_category": analysis_data.get("product_category", "unknown"),
            "current_angle": analysis_data.get("current_angle", "front-facing"),
            "visual_characteristics": analysis_data.get("visual_characteristics", {}),
            "colors": analysis_data.get("colors", []),
            "materials": analysis_data.get("materials", []),
            "positioning_recommendation": analysis_data.get("positioning_recommendation", ""),
            "composition_notes": analysis_data.get("composition_notes", ""),
            "needs_angle_regeneration": analysis_data.get("needs_angle_regeneration", False),
            "angle_regeneration_reason": analysis_data.get("angle_regeneration_reason"),
        }
        
        return {
            "product_analysis": product_analysis,
        }


# Singleton instance for import convenience
product_analyzer = ProductAnalyzerAgent()
