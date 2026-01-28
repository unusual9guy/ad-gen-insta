"""Agent 4: LinkedIn Text Generator - Uses Gemini to create LinkedIn post copy."""

from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from workflow.state import WorkflowState
from config.settings import settings


LINKEDIN_TEXT_SYSTEM = """You are an expert LinkedIn copywriter and social media strategist. Your task is to create engaging, professional LinkedIn post copy for product advertisements.

Your copy must:
- Be professional yet engaging and personable
- Highlight key product features and benefits
- Include a clear call-to-action
- Use appropriate hashtags (3-5 relevant ones)
- Be optimized for LinkedIn's algorithm (encourage engagement)
- Avoid being overly salesy - focus on value
- Be between 150-300 words for optimal engagement

LinkedIn best practices to follow:
- Start with a hook in the first line (this shows in preview)
- Use line breaks for readability
- Include a question or conversation starter
- End with a clear CTA"""


LINKEDIN_TEXT_PROMPT = """Create a professional LinkedIn post for the following product advertisement:

## Product Information
- **Product Name**: {product_name}
- **Product Type**: {product_type}
- **Product Category**: {product_category}
- **Key Colors**: {product_colors}
- **Materials**: {product_materials}

## Product Description
{composition_notes}

## Positioning
{positioning_recommendation}

## Requirements
- Write an engaging LinkedIn post (150-300 words)
- Start with an attention-grabbing hook
- Highlight 2-3 key features or benefits
- Include a professional call-to-action
- Add 3-5 relevant hashtags at the end
- Maintain a professional but approachable tone

Return ONLY the LinkedIn post text, ready to copy and paste. Do not include any additional commentary or explanations."""


class LinkedInTextGeneratorAgent(BaseAgent[str]):
    """
    Agent 4: Generates professional LinkedIn post copy using Gemini.
    
    This agent creates engaging LinkedIn post text based on the
    product analysis from Agent 1. Only executed for LinkedIn platform.
    """
    
    name = "LinkedInTextGenerator"
    description = "Generates LinkedIn post copy using Gemini"
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Gemini model for Agent 4."""
        api_key = settings.get_agent4_key()
        if api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.7,
                max_tokens=1024
            )
            self.logger.info("Agent 4 (LinkedIn Text) initialized with Gemini")
        else:
            self.logger.warning("No API key configured for Agent 4 (LinkedIn Text Generator)")
    
    def validate_inputs(self, state: WorkflowState) -> bool:
        """Validate that product analysis is present."""
        if not state.get("product_analysis"):
            self.logger.error("Product analysis is required")
            return False
        if state.get("platform") != "linkedin":
            self.logger.warning("LinkedIn text generation called for non-LinkedIn platform")
        return True
    
    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """
        Generate LinkedIn post copy based on product analysis.
        
        Args:
            state: Current workflow state with product_analysis
            
        Returns:
            Dictionary containing linkedin_post_text
        """
        if not self.llm:
            # Fallback to placeholder if no API key
            self.logger.warning("Using placeholder LinkedIn text - no API key configured")
            return self._generate_placeholder_text(state)
        
        analysis = state["product_analysis"]
        
        # Build the prompt
        user_prompt = LINKEDIN_TEXT_PROMPT.format(
            product_name=state["product_name"],
            product_type=analysis.get("product_type", "product"),
            product_category=analysis.get("product_category", "general"),
            product_colors=", ".join(analysis.get("colors", ["neutral"])),
            product_materials=", ".join(analysis.get("materials", ["quality materials"])),
            composition_notes=analysis.get("composition_notes", "A premium product designed for excellence."),
            positioning_recommendation=analysis.get("positioning_recommendation", "Premium positioning for discerning customers."),
        )
        
        # Create messages with system and user prompts
        messages = [
            SystemMessage(content=LINKEDIN_TEXT_SYSTEM),
            HumanMessage(content=user_prompt)
        ]
        
        # Generate response
        response = self.llm.invoke(messages)
        
        linkedin_text = response.content.strip()
        
        return {
            "linkedin_post_text": linkedin_text,
        }
    
    def _generate_placeholder_text(self, state: WorkflowState) -> dict[str, Any]:
        """Generate placeholder LinkedIn text when Gemini is unavailable."""
        analysis = state["product_analysis"]
        product_name = state["product_name"]
        
        materials = analysis.get("materials", ["premium materials"])
        colors = analysis.get("colors", ["elegant"])
        product_type = analysis.get("product_type", "product")
        category = analysis.get("product_category", "innovation")
        
        placeholder_text = f"""Introducing something special. 

Meet the {product_name} - a {product_type} that redefines what's possible in {category}.

Here's what sets it apart:

Crafted with {', '.join(materials[:2]) if materials else 'premium materials'}, every detail has been thoughtfully designed. The {', '.join(colors[:2]) if colors else 'refined'} aesthetic isn't just beautiful - it's functional.

{analysis.get('composition_notes', 'Built for those who appreciate quality and performance in equal measure.')}

Whether you're a professional looking to elevate your toolkit or someone who simply appreciates exceptional design, the {product_name} delivers on every front.

Ready to experience the difference? 

Drop a comment below or send me a message - I'd love to hear what features matter most to you.

#{category.replace(' ', '')} #ProductLaunch #Innovation #Quality #ProfessionalGrowth"""
        
        return {
            "linkedin_post_text": placeholder_text,
        }


# Singleton instance
linkedin_text_generator = LinkedInTextGeneratorAgent()
