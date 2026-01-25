"""Image processing service for logo overlay and manipulation."""

from PIL import Image, ImageOps, ImageEnhance
import io
from typing import Literal

from config.settings import settings


class ImageProcessor:
    """
    Handles image processing operations for the ad generator.
    
    Main responsibilities:
    - Logo processing (background removal, monochrome conversion, resizing)
    - Logo overlay on generated ads
    - Image resizing for different aspect ratios
    """
    
    def __init__(self):
        # Increased logo size to 18% for better visibility
        self.logo_size_percentage = 0.18  # 18% of ad width
        self.logo_padding_percentage = settings.logo_padding_percentage
    
    def remove_background(
        self,
        image: Image.Image,
        threshold: int = 240,
        white_only: bool = True
    ) -> Image.Image:
        """
        Remove white/light background from an image.
        
        Args:
            image: Input image
            threshold: Brightness threshold (0-255). Pixels brighter than this become transparent
            white_only: If True, only removes near-white pixels. If False, removes all light pixels.
            
        Returns:
            Image with transparent background
        """
        # Convert to RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        # Get pixel data
        pixels = image.load()
        width, height = image.size
        
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                
                if white_only:
                    # Check if pixel is near-white (all channels above threshold)
                    if r > threshold and g > threshold and b > threshold:
                        # Make transparent
                        pixels[x, y] = (r, g, b, 0)
                else:
                    # Check if pixel is light (average brightness above threshold)
                    brightness = (r + g + b) / 3
                    if brightness > threshold:
                        pixels[x, y] = (r, g, b, 0)
        
        return image
    
    def remove_background_smart(
        self,
        image: Image.Image,
        tolerance: int = 30
    ) -> Image.Image:
        """
        Smart background removal that detects the background color from corners.
        
        Args:
            image: Input image
            tolerance: Color tolerance for background detection
            
        Returns:
            Image with transparent background
        """
        # Convert to RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        pixels = image.load()
        width, height = image.size
        
        # Sample background color from corners
        corners = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1)
        ]
        
        # Get the most common corner color (likely the background)
        corner_colors = []
        for cx, cy in corners:
            r, g, b, a = pixels[cx, cy]
            corner_colors.append((r, g, b))
        
        # Use the average of corner colors as background reference
        avg_r = sum(c[0] for c in corner_colors) // 4
        avg_g = sum(c[1] for c in corner_colors) // 4
        avg_b = sum(c[2] for c in corner_colors) // 4
        
        # Remove pixels similar to background color
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                
                # Check if pixel is similar to background
                if (abs(r - avg_r) < tolerance and 
                    abs(g - avg_g) < tolerance and 
                    abs(b - avg_b) < tolerance):
                    pixels[x, y] = (r, g, b, 0)
        
        return image
    
    def process_logo(
        self,
        logo: Image.Image,
        target_width: int,
        make_monochrome: bool = False,
        remove_bg: bool = True
    ) -> Image.Image:
        """
        Process logo for overlay: remove background and resize.
        Keeps original colors for clarity.
        
        Args:
            logo: Original logo image
            target_width: Width of the target ad image
            make_monochrome: Whether to convert to grayscale (default False for clarity)
            remove_bg: Whether to remove the background
            
        Returns:
            Processed logo ready for overlay
        """
        # Calculate target logo size (percentage of ad width)
        logo_width = int(target_width * self.logo_size_percentage)
        
        # Ensure minimum size
        logo_width = max(logo_width, 150)
        
        # Maintain aspect ratio
        aspect_ratio = logo.width / logo.height if logo.height > 0 else 1
        logo_height = int(logo_width / aspect_ratio)
        
        # Ensure RGBA mode for transparency support BEFORE resizing
        if logo.mode != "RGBA":
            logo = logo.convert("RGBA")
        
        # Remove background BEFORE resizing for better quality
        if remove_bg:
            logo = self.remove_background(logo, threshold=240)
        
        # Resize logo with high quality AFTER background removal
        logo_resized = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        
        if make_monochrome:
            # Split channels
            r, g, b, a = logo_resized.split()
            
            # Convert RGB to grayscale
            rgb_image = Image.merge("RGB", (r, g, b))
            grayscale = ImageOps.grayscale(rgb_image)
            
            # Merge back with original alpha
            logo_mono = Image.merge("RGBA", (grayscale, grayscale, grayscale, a))
            return logo_mono
        
        return logo_resized
    
    def overlay_logo(
        self,
        ad_image: Image.Image,
        logo: Image.Image,
        position: Literal["top-right", "top-left", "bottom-right", "bottom-left"] = "top-right",
        remove_bg: bool = True
    ) -> Image.Image:
        """
        Overlay a logo onto the ad image.
        
        Args:
            ad_image: The generated ad image
            logo: The company logo
            position: Where to place the logo
            remove_bg: Whether to remove logo background
            
        Returns:
            Ad image with logo overlaid
        """
        # Ensure ad is RGBA for compositing
        ad_rgba = ad_image.convert("RGBA")
        
        # Process the logo (includes background removal, keeps original colors)
        processed_logo = self.process_logo(
            logo, 
            ad_image.width, 
            make_monochrome=False,  # Keep original colors for clarity
            remove_bg=remove_bg
        )
        
        # Calculate padding
        padding = int(ad_image.width * self.logo_padding_percentage)
        
        # Calculate position
        if position == "top-right":
            x = ad_image.width - processed_logo.width - padding
            y = padding
        elif position == "top-left":
            x = padding
            y = padding
        elif position == "bottom-right":
            x = ad_image.width - processed_logo.width - padding
            y = ad_image.height - processed_logo.height - padding
        elif position == "bottom-left":
            x = padding
            y = ad_image.height - processed_logo.height - padding
        else:
            x = ad_image.width - processed_logo.width - padding
            y = padding
        
        # Composite the logo onto the ad using alpha channel
        ad_rgba.paste(processed_logo, (x, y), processed_logo)
        
        return ad_rgba.convert("RGB")
    
    def resize_for_aspect_ratio(
        self,
        image: Image.Image,
        aspect_ratio: str,
        base_size: int = 1080
    ) -> Image.Image:
        """
        Resize/crop image to match target aspect ratio.
        
        Args:
            image: Source image
            aspect_ratio: Target aspect ratio (e.g., "1:1", "4:5", "1.91:1")
            base_size: Base dimension size
            
        Returns:
            Resized image matching the aspect ratio
        """
        # Parse aspect ratio
        if ":" in aspect_ratio:
            parts = aspect_ratio.split(":")
            width_ratio = float(parts[0])
            height_ratio = float(parts[1])
        else:
            # Handle decimal format like "1.91:1"
            width_ratio = float(aspect_ratio)
            height_ratio = 1.0
        
        # Calculate target dimensions
        if width_ratio >= height_ratio:
            target_width = base_size
            target_height = int(base_size * height_ratio / width_ratio)
        else:
            target_height = base_size
            target_width = int(base_size * width_ratio / height_ratio)
        
        # Use thumbnail to resize while maintaining aspect ratio
        image_copy = image.copy()
        image_copy.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Create canvas of exact target size
        canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        
        # Center the image on canvas
        x = (target_width - image_copy.width) // 2
        y = (target_height - image_copy.height) // 2
        
        if image_copy.mode == "RGBA":
            canvas.paste(image_copy, (x, y), image_copy)
        else:
            canvas.paste(image_copy, (x, y))
        
        return canvas


# Singleton instance
image_processor = ImageProcessor()
