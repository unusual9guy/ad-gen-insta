"""Agent 3: Ad Generator - Uses Google's Gemini image generation to create product ads."""

from typing import Any
from PIL import Image
import io
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from agents.base import BaseAgent
from workflow.state import WorkflowState
from services.image_processor import ImageProcessor
from config.settings import settings

load_dotenv()


class AdGeneratorAgent(BaseAgent[bytes]):
    """
    Agent 3: Generates product advertisement images using Google's Gemini image generation.
    
    This agent takes the image generation prompt from Agent 2 and
    creates a professional product ad image using models/gemini-3-pro-image-preview.
    It also handles logo overlay post-processing.
    """
    
    name = "AdGenerator"
    description = "Generates product ad images using Google's Gemini image generation"
    
    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor()
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Google GenAI client for Agent 3."""
        api_key = settings.get_agent3_key()
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.logger.info("Agent 3 (Ad Generator) initialized with Google GenAI")
        else:
            self.logger.warning("No API key configured for Agent 3 (Ad Generator)")
    
    def validate_inputs(self, state: WorkflowState) -> bool:
        """Validate that prompt and images are present."""
        if not state.get("image_generation_prompt"):
            self.logger.error("Image generation prompt is required")
            return False
        if not state.get("product_image"):
            self.logger.error("Product image is required")
            return False
        if not state.get("logo_image"):
            self.logger.error("Logo image is required")
            return False
        return True
    
    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """
        Generate the ad image with logo integrated by AI.
        
        Args:
            state: Current workflow state with image_generation_prompt
            
        Returns:
            Dictionary containing raw_generated_image and generated_ad_image
        """
        prompt = state["image_generation_prompt"]
        aspect_ratio = state["aspect_ratio"]
        product_image = Image.open(io.BytesIO(state["product_image"]))
        logo_image = Image.open(io.BytesIO(state["logo_image"]))
        
        # Get target dimensions
        target_size = self._get_target_size(aspect_ratio)
        
        # Enhance the prompt with critical instructions (now includes logo instruction)
        enhanced_prompt = self._enhance_prompt(prompt, aspect_ratio)
        
        # Try to generate the ad image with logo integrated by AI
        generated_image = await self._generate_image(
            prompt=enhanced_prompt,
            product_image=product_image,
            logo_image=logo_image,
            target_size=target_size,
        )
        
        # Save raw generated image
        raw_buffer = io.BytesIO()
        generated_image.save(raw_buffer, format="PNG")
        raw_generated_image = raw_buffer.getvalue()
        
        # ============================================================
        # MANUAL LOGO OVERLAY - COMMENTED OUT (using AI integration instead)
        # ============================================================
        # final_ad = self.image_processor.overlay_logo(
        #     ad_image=generated_image,
        #     logo=logo_image,
        #     position="top-right"
        # )
        # ============================================================
        
        # Use the AI-generated image directly (logo is integrated by AI)
        final_ad = generated_image
        
        # Convert final image to bytes
        final_buffer = io.BytesIO()
        final_ad.save(final_buffer, format="PNG")
        generated_ad_image = final_buffer.getvalue()
        
        return {
            "raw_generated_image": raw_generated_image,
            "generated_ad_image": generated_ad_image,
        }
    
    def _enhance_prompt(self, prompt: str, aspect_ratio: str) -> str:
        """Enhance the prompt with critical instructions for image generation."""
        
        # Map aspect ratio to dimensions
        dimensions = {
            "1:1": "1080x1080 pixels (square)",
            "4:5": "1080x1350 pixels (portrait)",
            "1.91:1": "1080x566 pixels (landscape)",
        }
        
        dimension_text = dimensions.get(aspect_ratio, "1080x1080 pixels (square)")
        
        critical_instructions = f"""
CRITICAL INSTRUCTIONS FOR IMAGE GENERATION:

1. ASPECT RATIO: Generate the image in EXACTLY {aspect_ratio} aspect ratio ({dimension_text}).

2. PROFESSIONAL QUALITY REQUIREMENTS:
   - This must look like it was created by a professional graphic designer
   - Study high-end brand ads from Apple, Nike, Dyson for inspiration
   - Use sophisticated color grading and subtle shadows
   - The overall feel should be premium, polished, and aspirational
   - Lighting should be soft, directional, and create depth
   - Shadows should be realistic and grounded

3. PRODUCT INTEGRATION - CRITICAL FOR REALISM:
   - The product MUST look like it BELONGS in the scene, not pasted on top
   - LIGHTING MATCH: The product's lighting must match the background
   - SHADOWS: Add realistic ground shadows that anchor the product
   - COLOR HARMONY: The product should share color tones with the environment
   - EDGE BLENDING: No harsh cutout edges - product should blend naturally

4. LOGO INTEGRATION - VERY IMPORTANT:
   - I am providing a company logo image along with the product image
   - Place this logo in the TOP-RIGHT CORNER of the final image
   - Make the logo BLACK/MONOCHROME (convert it to black color)
   - Make the logo SMALL - about 5-8% of image width (smaller than before, reduce by 40%)
   - Remove any white background from the logo - only show the logo graphic itself
   - The logo should look professionally integrated, not pasted on
   - Keep the logo subtle and elegant, not dominating the image

5. PRODUCT SHARPENING - ABSOLUTE TOP PRIORITY:
   - THIS IS MANDATORY: The product MUST be rendered with MAXIMUM SHARPNESS and CLARITY
   - ZERO BLUR TOLERANCE: Do NOT allow any blur, softness, or out-of-focus areas on the product
   - ENHANCE ALL EDGES: Every edge of the product must be crisp, defined, and razor-sharp
   - TEXTURE DETAIL: Show every texture, surface detail, and material quality in high definition
   - DEPTH OF FIELD: Keep the ENTIRE product in perfect focus - no bokeh or blur on any part
   - CONTRAST ENHANCEMENT: Increase local contrast on product edges for maximum definition
   - The product should look like a high-resolution professional product photo
   - If the input product image has ANY blur, you MUST recreate it with perfect sharpness
   - The product is the HERO of this image - it must be photographically perfect

6. KEEP IT CLEAN:
   - Do NOT add any other text or overlays
   - Focus on the product, professional background, and the logo placement

MAIN PROMPT:
{prompt}
"""
        return critical_instructions
    
    async def _generate_image(
        self,
        prompt: str,
        product_image: Image.Image,
        logo_image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """
        Generate the ad image using Google's Gemini image generation.
        """
        if not self.client:
            self.logger.warning("No client available, using placeholder")
            return self._create_placeholder_ad(product_image, target_size)
        
        try:
            # Prepare contents: prompt + product image + logo image
            # The logo is passed to the AI for integration
            contents = [prompt, product_image, logo_image]
            
            # Try Gemini 3 Pro first, fallback to Gemini 2.5 Flash
            model_name = "models/gemini-3-pro-image-preview"
            
            try:
                response = self.client.models.generate_content(
                    model="models/gemini-3-pro-image-preview",
                    contents=contents,
                )
                self.logger.info("Generated image with gemini-3-pro-image-preview")
            except Exception as pro_error:
                self.logger.warning(f"Gemini 3 Pro failed: {pro_error}, trying fallback...")
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=contents,
                    )
                    model_name = "gemini-2.5-flash-image"
                    self.logger.info("Generated image with gemini-2.5-flash-image")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
                    raise pro_error
            
            # Process the response - look for image data
            generated_image = None
            result_text = ""
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text is not None:
                    result_text += part.text
                elif hasattr(part, 'inline_data') and part.inline_data is not None:
                    # Found the generated image
                    image_data = part.inline_data.data
                    generated_image = Image.open(io.BytesIO(image_data))
                    self.logger.info(f"Successfully extracted generated image: {generated_image.size}")
            
            if generated_image:
                # Resize to target size if needed
                if generated_image.size != target_size:
                    generated_image = generated_image.resize(target_size, Image.Resampling.LANCZOS)
                return generated_image
            else:
                self.logger.warning("No image in response, using placeholder")
                self.logger.debug(f"Response text: {result_text[:500] if result_text else 'None'}")
                return self._create_placeholder_ad(product_image, target_size)
                
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return self._create_placeholder_ad(product_image, target_size)
    
    def _get_target_size(self, aspect_ratio: str) -> tuple[int, int]:
        """Get target dimensions for the given aspect ratio."""
        base_size = 1080  # Base size for social media
        
        if aspect_ratio == "1:1":
            return (base_size, base_size)
        elif aspect_ratio == "4:5":
            return (base_size, int(base_size * 5 / 4))
        elif aspect_ratio == "1.91:1":
            return (base_size, int(base_size / 1.91))
        else:
            return (base_size, base_size)
    
    def _create_placeholder_ad(
        self,
        product_image: Image.Image,
        target_size: tuple[int, int]
    ) -> Image.Image:
        """
        Create a placeholder ad image when AI generation is unavailable.
        """
        from PIL import ImageDraw, ImageFilter
        
        width, height = target_size
        
        # Create a gradient background
        background = Image.new("RGB", (width, height), (245, 245, 250))
        draw = ImageDraw.Draw(background)
        
        # Add a subtle gradient
        for y in range(height):
            gray = int(245 - (y / height) * 20)
            draw.line([(0, y), (width, y)], fill=(gray, gray, gray + 5))
        
        # Prepare product image
        product_image = product_image.convert("RGBA")
        
        # Calculate size to fit product (max 65% of frame)
        max_product_width = int(width * 0.65)
        max_product_height = int(height * 0.65)
        
        product_ratio = product_image.width / product_image.height
        
        if product_ratio > max_product_width / max_product_height:
            new_width = max_product_width
            new_height = int(new_width / product_ratio)
        else:
            new_height = max_product_height
            new_width = int(new_height * product_ratio)
        
        product_resized = product_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        # Add subtle shadow under product
        shadow = Image.new("RGBA", (new_width + 20, new_height + 20), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.ellipse(
            [10, new_height - 10, new_width + 10, new_height + 30],
            fill=(0, 0, 0, 30)
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
        
        # Center position
        x = (width - new_width) // 2
        y = (height - new_height) // 2
        
        # Composite shadow and product
        background = background.convert("RGBA")
        background.paste(shadow, (x - 10, y + 10), shadow)
        background.paste(product_resized, (x, y), product_resized)
        
        return background.convert("RGB")


# Singleton instance
ad_generator = AdGeneratorAgent()
