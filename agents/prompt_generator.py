"""Agent 2: Prompt Generator - Uses Gemini to create image generation prompts."""

from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from workflow.state import WorkflowState
from config.settings import settings
from config.templates import get_template_for_product, BackgroundTemplate, BACKGROUND_TEMPLATES


PROMPT_GENERATION_SYSTEM = """You are an expert creative director specializing in product photography and advertising. Your task is to create detailed, effective image generation prompts that will produce stunning product advertisements.

You will receive:
1. Product analysis data (type, colors, materials, positioning recommendations)
2. A background template with aesthetic guidelines
3. Target aspect ratio

Your prompt must:
- Seamlessly integrate the product into the background aesthetic
- Maintain the template's visual consistency while adapting it to the specific product
- Include specific lighting directions that complement both product and background
- Specify exact composition and framing for the given aspect ratio
- Focus on commercial quality that would work for real advertising
- NEVER describe the product itself in ways that would change it - the product image will be composited in

Output a single, cohesive image generation prompt that an AI image generator can use to create the perfect background and composition for this product advertisement."""


PROMPT_GENERATION_USER = """Create an image generation prompt for the following product advertisement:

## Product Information
- **Product Name**: {product_name}
- **Product Type**: {product_type}
- **Product Category**: {product_category}
- **Product Colors**: {product_colors}
- **Product Materials**: {product_materials}
- **Positioning Recommendation**: {positioning_recommendation}
- **Composition Notes**: {composition_notes}

## Background Template: {template_name}
{template_description}

### Template Guidelines:
- **Base Style**: {template_base_prompt}
- **Color Palette**: {template_colors}
- **Lighting**: {template_lighting}
- **Mood**: {template_mood}
- **Surface Materials**: {template_surfaces}
- **Props**: {template_props}

## Target Format
- **Platform**: {platform}
- **Aspect Ratio**: {aspect_ratio}

## Instructions
Generate a detailed image generation prompt that:
1. Creates the perfect background environment following the template guidelines
2. Leaves clear space for the product to be composited based on the positioning recommendation
3. Adapts the template's aesthetic to complement this specific product's colors and style
4. Specifies exact composition for {aspect_ratio} aspect ratio
5. Includes professional lighting setup that will match the product

Return ONLY the image generation prompt, no explanations or additional text. The prompt should be 2-4 paragraphs of detailed, specific instructions for the image generator."""


class PromptGeneratorAgent(BaseAgent[str]):
    """
    Agent 2: Generates optimized image generation prompts using Gemini.
    
    This agent takes the product analysis from Agent 1 and creates
    a detailed prompt for image generation that will produce a professional
    product advertisement with consistent brand aesthetics.
    """
    
    name = "PromptGenerator"
    description = "Generates image generation prompts using Gemini"
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini model for Agent 2."""
        api_key = settings.get_agent2_key() or settings.google_api_key
        if api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.7,
                max_tokens=1024
            )
            self.logger.info("Agent 2 (Prompt Generator) initialized with Gemini")
        else:
            self.logger.warning("No API key configured for Agent 2 (Prompt Generator)")
    
    def validate_inputs(self, state: WorkflowState) -> bool:
        """Validate that product analysis is present."""
        if not state.get("product_analysis"):
            self.logger.error("Product analysis is required")
            return False
        return True
    
    def _select_template(self, state: WorkflowState) -> BackgroundTemplate:
        """Select the most appropriate background template for the product."""
        selected_category = state.get("selected_category", "others")
        
        self.logger.info(f"[DEBUG] selected_category from state: '{selected_category}'")
        self.logger.info(f"[DEBUG] Available templates: {list(BACKGROUND_TEMPLATES.keys())}")
        
        # If user selected a specific category, use that template directly
        if selected_category != "others" and selected_category in BACKGROUND_TEMPLATES:
            template = BACKGROUND_TEMPLATES[selected_category]
            self.logger.info(f"Using user-selected category template: {selected_category} -> {template['name']}")
            return template
        
        # Otherwise, auto-detect from product analysis
        analysis = state["product_analysis"]
        product_category = analysis.get("product_category", "general")
        self.logger.info(f"Auto-detecting template from product category: {product_category}")
        
        # Get template based on product category
        return get_template_for_product(product_category)
    
    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """
        Generate an image generation prompt based on product analysis.
        
        Args:
            state: Current workflow state with product_analysis
            
        Returns:
            Dictionary containing image_generation_prompt and background_template_used
        """
        if not self.llm:
            # Fallback to placeholder if no API key
            self.logger.warning("Using placeholder prompt - no API key configured")
            return self._generate_placeholder_prompt(state)
        
        analysis = state["product_analysis"]
        template = self._select_template(state)
        
        # Build the user prompt with all context
        user_prompt = PROMPT_GENERATION_USER.format(
            product_name=state["product_name"],
            product_type=analysis.get("product_type", "product"),
            product_category=analysis.get("product_category", "general"),
            product_colors=", ".join(analysis.get("colors", ["neutral"])),
            product_materials=", ".join(analysis.get("materials", ["mixed"])),
            positioning_recommendation=analysis.get("positioning_recommendation", "centered"),
            composition_notes=analysis.get("composition_notes", ""),
            template_name=template["name"],
            template_description=template["description"],
            template_base_prompt=template["base_prompt"],
            template_colors=", ".join(template["color_palette"]),
            template_lighting=template["lighting"],
            template_mood=template["mood"],
            template_surfaces=", ".join(template["surface_materials"]),
            template_props=", ".join(template["props"]) if template["props"] else "none - minimal composition",
            platform=state["platform"].title(),
            aspect_ratio=state["aspect_ratio"],
        )
        
        # Create messages with system and user prompts
        messages = [
            SystemMessage(content=PROMPT_GENERATION_SYSTEM),
            HumanMessage(content=user_prompt)
        ]
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Extract the generated prompt
        generated_prompt = response.content.strip()
        
        return {
            "image_generation_prompt": generated_prompt,
            "background_template_used": template["name"],
        }
    
    def _generate_placeholder_prompt(self, state: WorkflowState) -> dict[str, Any]:
        """Generate a placeholder prompt when no API is available."""
        analysis = state["product_analysis"]
        template = self._select_template(state)
        product_name = state["product_name"]
        aspect_ratio = state["aspect_ratio"]
        
        placeholder_prompt = f"""Professional product advertisement photograph for {product_name}.

{template['base_prompt']}

The composition is optimized for {aspect_ratio} aspect ratio with the product positioned according to: {analysis.get('positioning_recommendation', 'center frame')}.

Color palette harmonizes with the product's {', '.join(analysis.get('colors', ['neutral']))} tones, using {', '.join(template['color_palette'][:3])} in the background.

Lighting setup: {template['lighting']}

The mood conveys: {template['mood']}

Surface and environment: {', '.join(template['surface_materials'])}

Commercial-quality product photography with exceptional attention to detail, professional color grading, and premium aesthetic suitable for {state['platform'].title()} advertising."""
        
        return {
            "image_generation_prompt": placeholder_prompt,
            "background_template_used": template["name"],
        }


# Singleton instance
prompt_generator = PromptGeneratorAgent()
