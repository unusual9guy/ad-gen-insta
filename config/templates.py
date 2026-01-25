"""Background aesthetic templates for consistent brand visuals."""

from typing import TypedDict


class BackgroundTemplate(TypedDict):
    """Structure for background aesthetic templates."""
    name: str
    description: str
    base_prompt: str
    color_palette: list[str]
    lighting: str
    mood: str
    surface_materials: list[str]
    props: list[str]


# Default brand-consistent background templates
BACKGROUND_TEMPLATES: dict[str, BackgroundTemplate] = {
    "minimal_studio": {
        "name": "Minimal Studio",
        "description": "Clean, professional studio setup with soft gradients",
        "base_prompt": """Professional product photography in a minimal studio setting.
Clean, uncluttered background with a soft gradient transitioning from light to slightly darker tones.
Soft diffused studio lighting creating gentle shadows that give depth without harsh contrasts.
The product is the clear focal point with subtle environmental lighting enhancing its features.
High-end commercial photography aesthetic with crisp details and professional color grading.""",
        "color_palette": [
            "soft white",
            "light gray",
            "subtle warm gray gradient",
            "pearl white",
            "off-white cream"
        ],
        "lighting": "Soft diffused studio lighting with gentle rim light and subtle fill, creating professional depth without harsh shadows",
        "mood": "Clean, professional, premium, trustworthy",
        "surface_materials": ["seamless paper backdrop", "subtle matte surface"],
        "props": []  # Minimal - no props
    },
    
    "lifestyle_modern": {
        "name": "Modern Lifestyle",
        "description": "Contemporary lifestyle setting with natural elements",
        "base_prompt": """Modern lifestyle product photography with natural, organic elements.
Warm, inviting atmosphere with natural materials visible in the background.
Natural window light creating soft, directional illumination with warm undertones.
Subtle lifestyle context that complements the product without distracting from it.
Contemporary aesthetic with clean lines and thoughtful composition.""",
        "color_palette": [
            "warm neutrals",
            "natural wood tones",
            "soft beige",
            "warm white",
            "muted sage green"
        ],
        "lighting": "Natural window light with warm undertones, soft shadows, golden hour quality",
        "mood": "Warm, inviting, authentic, aspirational",
        "surface_materials": ["light wood surface", "natural linen", "marble accent"],
        "props": ["subtle greenery", "natural textures"]
    },
    
    "tech_premium": {
        "name": "Tech Premium",
        "description": "Sleek, futuristic aesthetic for technology products",
        "base_prompt": """Premium technology product photography with sleek, modern aesthetic.
Dark, sophisticated background with subtle metallic or glass reflections.
Dramatic lighting with precise highlights accentuating product curves and surfaces.
High-tech atmosphere suggesting innovation and premium quality.
Sharp, precise commercial photography with exceptional detail rendering.""",
        "color_palette": [
            "deep charcoal",
            "midnight blue",
            "subtle metallic silver",
            "matte black",
            "accent of electric blue"
        ],
        "lighting": "Dramatic rim lighting with controlled highlights, subtle gradient spots, high contrast with preserved shadow detail",
        "mood": "Innovative, premium, cutting-edge, sophisticated",
        "surface_materials": ["brushed metal surface", "dark glass", "matte finish"],
        "props": []  # Clean tech aesthetic
    },
    
    "nature_organic": {
        "name": "Nature & Organic",
        "description": "Natural, eco-friendly aesthetic with organic elements",
        "base_prompt": """Organic, nature-inspired product photography with eco-conscious aesthetic.
Natural textures and materials creating an authentic, sustainable atmosphere.
Soft, diffused natural lighting suggesting outdoor or greenhouse environment.
Earthy tones and botanical elements complementing the product naturally.
Authentic, honest commercial photography with artisanal quality.""",
        "color_palette": [
            "sage green",
            "natural terracotta",
            "warm sand",
            "forest green accents",
            "cream white"
        ],
        "lighting": "Soft diffused daylight, greenhouse-style lighting with gentle leaf shadows",
        "mood": "Natural, sustainable, authentic, wholesome",
        "surface_materials": ["raw wood", "natural stone", "woven materials", "dried botanicals"],
        "props": ["subtle plant elements", "natural textures", "organic materials"]
    },
    
    "luxury_elegance": {
        "name": "Luxury Elegance",
        "description": "Sophisticated, high-end aesthetic for luxury products",
        "base_prompt": """Luxurious, elegant product photography with refined sophistication.
Rich, deep background colors suggesting opulence and exclusivity.
Dramatic yet refined lighting creating depth and highlighting premium materials.
Subtle hints of luxury materials like velvet, marble, or gold accents.
Editorial-quality commercial photography with impeccable attention to detail.""",
        "color_palette": [
            "deep burgundy",
            "rich navy",
            "champagne gold accents",
            "cream white",
            "soft blush pink"
        ],
        "lighting": "Refined dramatic lighting with soft gradients, subtle rim lights, jewel-like highlights",
        "mood": "Luxurious, exclusive, refined, aspirational",
        "surface_materials": ["velvet fabric", "marble surface", "metallic gold accents"],
        "props": ["subtle fabric draping", "reflective surfaces"]
    },
    
    "home_decor_luxury": {
        "name": "Home Decor / Luxury Home Decor",
        "description": "Warm, natural aesthetic with dried botanicals for home decor products",
        "base_prompt": """Professional product photography background, eye-level shot, placed on a textured beige sandstone surface, background features blurred dried wheat stalks and small dried floral elements, soft warm natural window light coming from the left, shallow depth of field, strong bokeh effect in background, neutral earth tones, cream and tan color palette, minimalist, authentic texture, 8k resolution, photorealistic, no product in center, empty space for product placement.""",
        "color_palette": [
            "beige",
            "cream",
            "tan",
            "neutral earth tones",
            "warm sand"
        ],
        "lighting": "Soft warm natural window light coming from the left, shallow depth of field",
        "mood": "Warm, natural, minimalist, authentic, cozy",
        "surface_materials": ["textured beige sandstone surface"],
        "props": ["dried wheat stalks", "small dried floral elements"]
    }
}


def get_template(template_name: str) -> BackgroundTemplate:
    """
    Get a background template by name.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        BackgroundTemplate dictionary
        
    Raises:
        KeyError: If template name not found
    """
    if template_name not in BACKGROUND_TEMPLATES:
        raise KeyError(f"Template '{template_name}' not found. Available: {list(BACKGROUND_TEMPLATES.keys())}")
    return BACKGROUND_TEMPLATES[template_name]


def get_template_for_product(product_category: str) -> BackgroundTemplate:
    """
    Suggest an appropriate template based on product category.
    
    Args:
        product_category: The product category from analysis
        
    Returns:
        Suggested BackgroundTemplate
    """
    category_lower = product_category.lower()
    
    # Map categories to templates
    if any(term in category_lower for term in ["tech", "electronic", "gadget", "device", "computer", "phone"]):
        return BACKGROUND_TEMPLATES["tech_premium"]
    elif any(term in category_lower for term in ["organic", "natural", "eco", "sustainable", "plant", "food"]):
        return BACKGROUND_TEMPLATES["nature_organic"]
    elif any(term in category_lower for term in ["luxury", "jewelry", "watch", "fashion", "designer", "premium"]):
        return BACKGROUND_TEMPLATES["luxury_elegance"]
    elif any(term in category_lower for term in ["home", "decor", "furniture", "candle", "vase", "interior"]):
        return BACKGROUND_TEMPLATES["home_decor_luxury"]
    elif any(term in category_lower for term in ["lifestyle"]):
        return BACKGROUND_TEMPLATES["lifestyle_modern"]
    else:
        # Default to minimal studio for unknown categories
        return BACKGROUND_TEMPLATES["minimal_studio"]


def list_templates() -> list[str]:
    """Get list of available template names."""
    return list(BACKGROUND_TEMPLATES.keys())
