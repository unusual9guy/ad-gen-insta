"""Agent 3: Ad Generator - Uses Google's Gemini image generation to create product ads."""

from typing import Any
from PIL import Image
import io
import os
import time
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
    
    def _resize_for_api(self, image: Image.Image, max_dimension: int = 768) -> Image.Image:
        """
        Resize image to reduce API payload and prevent timeouts.
        
        Args:
            image: PIL Image to resize
            max_dimension: Maximum width or height (default 768px)
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        
        # Only resize if larger than max_dimension
        if width <= max_dimension and height <= max_dimension:
            return image
        
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height} for API")
        return resized
    
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
        additional_comments = state.get("additional_comments")
        product_image = Image.open(io.BytesIO(state["product_image"]))
        logo_image = Image.open(io.BytesIO(state["logo_image"]))
        
        # Get target dimensions
        target_size = self._get_target_size(aspect_ratio)
        
        # Enhance the prompt with critical instructions (skip for pomelli/rakhi modes)
        if "POMELLI MODE" in prompt:
            enhanced_prompt = prompt  # Pomelli prompt is self-contained
        elif "RAKHI PHOTOSHOOT" in prompt:
            enhanced_prompt = prompt  # Rakhi prompt is self-contained
        else:
            enhanced_prompt = self._enhance_prompt(prompt, aspect_ratio, additional_comments)
        
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
    
    def _enhance_prompt(self, prompt: str, aspect_ratio: str, additional_comments: str = None) -> str:
        """Enhance the prompt with critical instructions for image generation."""
        from typing import Optional
        
        # Map aspect ratio to dimensions
        dimensions = {
            "1:1": "1080x1080 pixels (square)",
            "4:5": "1080x1350 pixels (portrait)",
            "1.91:1": "1080x566 pixels (landscape)",
        }
        
        dimension_text = dimensions.get(aspect_ratio, "1080x1080 pixels (square)")
        
        # Build additional instructions section if provided
        additional_section = ""
        if additional_comments and additional_comments.strip():
            additional_section = f"""
7. ADDITIONAL USER INSTRUCTIONS - IMPORTANT:
   {additional_comments}
   - Follow these additional instructions carefully while maintaining overall quality
"""
        
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
{additional_section}
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
        Includes image resizing and retry logic to prevent timeouts.
        """
        if not self.client:
            self.logger.warning("No client available, using placeholder")
            return self._create_placeholder_ad(product_image, target_size)
        
        # Resize images to reduce payload and prevent timeouts
        product_resized = self._resize_for_api(product_image, max_dimension=768)
        
        # For Rakhi mode, only send the rakhi image (no logo needed)
        if "RAKHI PHOTOSHOOT" in prompt:
            contents = [prompt, product_resized]
        else:
            logo_resized = self._resize_for_api(logo_image, max_dimension=256)
            # Prepare contents: prompt + resized product image + resized logo image
            contents = [prompt, product_resized, logo_resized]
        
        # Retry configuration
        max_retries = 3
        retry_delays = [5, 10, 20]  # Exponential backoff in seconds
        
        models_to_try = [
            "models/gemini-3-pro-image-preview",
            "gemini-2.5-flash-image"
        ]
        
        last_error = None
        
        for attempt in range(max_retries):
            for model_name in models_to_try:
                try:
                    self.logger.info(f"Attempt {attempt + 1}/{max_retries}: Trying {model_name}...")
                    
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                    )
                    
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
                            self.logger.info(f"Successfully generated image with {model_name}: {generated_image.size}")
                    
                    if generated_image:
                        # Resize to target size if needed
                        if generated_image.size != target_size:
                            generated_image = generated_image.resize(target_size, Image.Resampling.LANCZOS)
                        return generated_image
                    else:
                        self.logger.warning(f"No image in response from {model_name}")
                        self.logger.debug(f"Response text: {result_text[:500] if result_text else 'None'}")
                        continue  # Try next model
                        
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    self.logger.warning(f"{model_name} failed: {error_str}")
                    
                    # If it's a 503 timeout, continue to next model or retry
                    if "503" in error_str or "UNAVAILABLE" in error_str:
                        continue
                    else:
                        # For other errors, skip to next model
                        continue
            
            # All models failed for this attempt, wait before retrying
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                self.logger.info(f"All models failed. Waiting {delay}s before retry {attempt + 2}...")
                time.sleep(delay)
        
        # All retries exhausted
        self.logger.error(f"Image generation failed after {max_retries} attempts. Last error: {last_error}")
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
