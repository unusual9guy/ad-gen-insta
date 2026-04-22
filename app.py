"""
Multi-Agent Product Ad Generator
================================
A Streamlit application that generates professional Instagram and LinkedIn 
product ads using a multi-agent LangGraph workflow.
"""

import streamlit as st
from PIL import Image
import io
import zipfile
import os
from typing import Optional
import logging
from datetime import datetime

from services.image_processor import ImageProcessor
from config.settings import settings
from config.templates import BACKGROUND_TEMPLATES
from workflow import create_initial_state
from workflow.nodes import (
    agent1_product_analyzer_sync,
    agent2_prompt_generator_sync,
    agent3_ad_generator_sync,
    agent4_linkedin_text_sync,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        # Step 1-2: Image uploads (first)
        "product_image": None,
        "product_image_preview": None,
        "logo_image": None,
        "logo_image_preview": None,
        # Step 3-4: Platform config (after uploads)
        "platform": "instagram",
        "aspect_ratio": "1:1",
        # Step 5: Product details
        "product_name": "",
        # Step 6-7: Category and additional comments
        "selected_category": "others",
        "additional_comments": "",
        # Workflow results
        "workflow_state": None,
        "generated_ad": None,
        "linkedin_text": None,
        # UI state
        "generation_in_progress": False,
        "error_message": None,
        "show_history": False,
        "current_step": 0,  # For progress tracking
        # Bulk Rakhi mode
        "bulk_rakhi_images": [],       # list of (bytes, PIL.Image) tuples
        "bulk_rakhi_results": [],      # list of PIL.Image (generated ads)
        "bulk_rakhi_errors": [],       # list of error strings (None if success)
        "bulk_rakhi_progress": 0,      # current count completed
        "bulk_rakhi_total": 0,         # total images in batch
        "bulk_mode_active": False,     # whether bulk mode is in progress
        "bulk_mode_done": False,       # whether bulk mode finished
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_uploaded_image(uploaded_file) -> Optional[tuple[bytes, Image.Image]]:
    """Process an uploaded image file and return bytes and PIL Image."""
    if uploaded_file is None:
        return None
    
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    return image_bytes, image


def get_aspect_ratio_options(platform: str) -> list[str]:
    """Get available aspect ratios based on selected platform."""
    if platform == "instagram" or platform == "pomelli" or platform == "rakhi":
        return settings.instagram_aspect_ratios
    else:
        return settings.linkedin_aspect_ratios


def render_sidebar():
    """Render the sidebar configuration."""
    with st.sidebar:
        # Header in sidebar
        st.markdown("""
            <div style="margin-bottom: 2rem;">
                <h2 style="font-family: var(--font-display); font-size: 1.5rem; font-weight: 800; color: var(--accent-primary); margin-bottom: 0;">ADGEN</h2>
                <div style="font-size: 0.8rem; font-family: var(--font-body); color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase;">Studio</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<h2 style='font-family: var(--font-display); font-size: 1.5rem; margin-bottom: 1.5rem; font-weight: 400; color: var(--text-main);'>Configuration</h2>", unsafe_allow_html=True)
        
        # Step 1: Platform Selection (moved to top)
        st.markdown("### 1. Platform")
        platform = st.radio(
            "Select target platform",
            options=["instagram", "linkedin", "pomelli", "rakhi"],
            format_func=lambda x: {"instagram": "Instagram", "linkedin": "LinkedIn", "pomelli": "Pomelli", "rakhi": "Rakhi"}[x],
            key="platform_selector",
            horizontal=True
        )
        st.session_state.platform = platform
        
        is_pomelli = platform == "pomelli"
        is_rakhi = platform == "rakhi"
        
        st.divider()
        
        # Step 2: Product Image Upload (or Rakhi Images for rakhi mode)
        if is_rakhi:
            st.markdown("### 2. Rakhi Images")
            st.caption("Upload 1–20 rakhi photos for bulk photoshoot generation")
            rakhi_files = st.file_uploader(
                "Upload your rakhi photos",
                type=["png", "jpg", "jpeg", "webp"],
                key="product_uploader",
                accept_multiple_files=True,
                help="Upload up to 20 clear photos of your rakhis for photoshoot-level enhancement"
            )
            
            if rakhi_files:
                if len(rakhi_files) > 20:
                    st.warning(f"⚠️ Maximum 20 images allowed. Only the first 20 of {len(rakhi_files)} will be processed.")
                    rakhi_files = rakhi_files[:20]
                
                # Process and store all uploaded images
                bulk_images = []
                for f in rakhi_files:
                    result = process_uploaded_image(f)
                    if result:
                        bulk_images.append(result)
                
                st.session_state.bulk_rakhi_images = bulk_images
                # Also set product_image to the first one for compatibility
                if bulk_images:
                    st.session_state.product_image = bulk_images[0][0]
                    st.session_state.product_image_preview = bulk_images[0][1]
                
                # Show thumbnail grid
                st.markdown(f"**{len(bulk_images)} image(s) ready**")
                cols = st.columns(4)
                for idx, (img_bytes, img_pil) in enumerate(bulk_images):
                    with cols[idx % 4]:
                        st.image(img_pil, caption=f"Rakhi {idx + 1}", use_container_width=True)
            else:
                st.session_state.bulk_rakhi_images = []
                st.session_state.product_image = None
                st.session_state.product_image_preview = None
        else:
            st.markdown("### 2. Product Image")
            product_file = st.file_uploader(
                "Upload your product image",
                type=["png", "jpg", "jpeg", "webp"],
                key="product_uploader",
                help="Upload a clear image of your product"
            )
            
            if product_file:
                result = process_uploaded_image(product_file)
                if result:
                    st.session_state.product_image = result[0]
                    st.session_state.product_image_preview = result[1]
                    st.image(result[1], caption="Product Preview", use_container_width=True)
        
        st.divider()
        
        # Step 3: Logo Upload
        st.markdown("### 3. Company Logo")
        logo_file = st.file_uploader(
            "Upload your company logo",
            type=["png", "jpg", "jpeg", "webp"],
            key="logo_uploader",
            help="Logo will be converted to black and placed in top-right corner"
        )
        
        if logo_file:
            result = process_uploaded_image(logo_file)
            if result:
                st.session_state.logo_image = result[0]
                st.session_state.logo_image_preview = result[1]
                st.image(result[1], caption="Logo Preview", use_container_width=True)
        
        st.divider()
        
        # Only show these fields for standard ad modes (not pomelli or rakhi)
        if not is_pomelli and not is_rakhi:
            # Step 4: Aspect Ratio (options depend on platform)
            aspect_options = get_aspect_ratio_options(platform)
            
            # Reset aspect ratio if current selection not valid for new platform
            if st.session_state.aspect_ratio not in aspect_options:
                st.session_state.aspect_ratio = aspect_options[0]
            
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=aspect_options,
                index=aspect_options.index(st.session_state.aspect_ratio),
                key="aspect_selector"
            )
            st.session_state.aspect_ratio = aspect_ratio
            
            st.divider()
            
            # Step 5: Product Name
            st.markdown("### 4. Product Details")
            product_name = st.text_input(
                "Product Name",
                value=st.session_state.product_name,
                placeholder="e.g., Premium Wireless Headphones",
                key="product_name_input"
            )
            st.session_state.product_name = product_name
            
            # Step 6: Category Selection
            st.markdown("### 5. Product Category")
            
            # Build category options from templates
            category_options = {
                "others": "Others (Auto-detect)",
            }
            for key, template in BACKGROUND_TEMPLATES.items():
                category_options[key] = template["name"]
            
            selected_category = st.selectbox(
                "Select Category",
                options=list(category_options.keys()),
                format_func=lambda x: category_options[x],
                index=list(category_options.keys()).index(st.session_state.selected_category),
                key="category_selector",
                help="Select the category that best matches your product. This determines the background style."
            )
            st.session_state.selected_category = selected_category
            
            # Step 7: Additional Comments
            st.markdown("### 6. Additional Instructions")
            additional_comments = st.text_area(
                "Additional Comments (Optional)",
                value=st.session_state.additional_comments,
                placeholder="e.g., Add cutlery next to my product - cutlery holder\nAdd an elegant whiskey glass on top - coaster",
                height=100,
                key="additional_comments_input",
                help="Extra instructions for the AI to customize the ad (props, styling, etc.)"
            )
            st.session_state.additional_comments = additional_comments
            
            st.divider()
        elif is_pomelli:
            # Pomelli mode info
            st.info("Pomelli mode: Your photo will be kept as-is with a black logo overlay in the top-right corner.")
        elif is_rakhi:
            # Rakhi mode info
            st.info("🪷 Rakhi mode: Upload a raw rakhi photo and your company logo. We'll create a stunning photoshoot-level image with your black logo integrated.")
        
        # Generate Button
        if is_rakhi:
            bulk_count = len(st.session_state.bulk_rakhi_images)
            can_generate = all([
                bulk_count > 0,
                st.session_state.logo_image,
            ])
        elif is_pomelli:
            can_generate = all([
                st.session_state.product_image,
                st.session_state.logo_image,
            ])
        else:
            can_generate = all([
                st.session_state.product_image,
                st.session_state.logo_image,
                st.session_state.product_name.strip()
            ])
        
        if is_rakhi:
            bulk_count = len(st.session_state.bulk_rakhi_images)
            btn_label = f"✨ Create Rakhi Photoshoot ({bulk_count} image{'s' if bulk_count != 1 else ''})" if bulk_count > 0 else "✨ Create Rakhi Photoshoot"
        elif is_pomelli:
            btn_label = "Apply Logo"
        else:
            btn_label = "Generate Ad"
        
        generate_button = st.button(
            btn_label,
            type="primary",
            disabled=not can_generate or st.session_state.generation_in_progress or st.session_state.bulk_mode_active,
            use_container_width=True
        )
        
        if generate_button:
            st.session_state.generation_in_progress = True
            st.session_state.current_step = 0
            st.session_state.generated_ad = None
            st.session_state.linkedin_text = None
            if is_rakhi:
                st.session_state.bulk_mode_active = True
                st.session_state.bulk_mode_done = False
                st.session_state.bulk_rakhi_results = []
                st.session_state.bulk_rakhi_errors = []
                st.session_state.bulk_rakhi_progress = 0
                st.session_state.bulk_rakhi_total = len(st.session_state.bulk_rakhi_images)
            st.rerun()
        
        if not can_generate:
            missing = []
            if is_rakhi:
                if len(st.session_state.bulk_rakhi_images) == 0:
                    missing.append("rakhi image(s)")
            elif not st.session_state.product_image:
                missing.append("product image")
            if not st.session_state.logo_image:
                missing.append("company logo")
            if not is_pomelli and not is_rakhi and not st.session_state.product_name.strip():
                missing.append("product name")
            
            st.caption(f"Missing: {', '.join(missing)}")
        
        # History toggle
        st.divider()
        if st.button("View History", use_container_width=True):
            st.session_state.show_history = not st.session_state.show_history


def render_main_content():
    """Render the main content area with results."""
    # Header
    st.markdown("""
        <div class="animated-hero" style="padding: 2rem 0; margin-bottom: 2rem; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem;">
                <div style="width: 8px; height: 8px; background: #FFFFFF; border-radius: 50%;"></div>
                <span style="font-family: 'Manrope', sans-serif; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.75rem; color: #AAAAAA;">Adgen Studio • System Online</span>
            </div>
            <h1 style="font-size: clamp(2.5rem, 4vw, 4rem) !important; margin-bottom: 0 !important; line-height: 1; letter-spacing: -0.02em; color: #FFFFFF; font-weight: 400 !important;">
                Orchestrate <span style="font-style: italic; color: #AAAAAA;">Professional</span> Campaigns
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='font-family: var(--font-display); font-size: 1.5rem; margin-bottom: 1.5rem; font-weight: 400; color: var(--text-main);'>Preview & Results</h2>", unsafe_allow_html=True)
    # Show generation progress or results
    if st.session_state.generation_in_progress:
        if st.session_state.platform == "rakhi" and st.session_state.bulk_mode_active:
            render_bulk_rakhi_progress()
        elif st.session_state.platform == "rakhi":
            render_rakhi_progress()
        elif st.session_state.platform == "pomelli":
            render_pomelli_progress()
        else:
            render_generation_progress()
    elif st.session_state.bulk_mode_done:
        render_bulk_rakhi_results()
    elif st.session_state.generated_ad:
        render_results()
    elif st.session_state.show_history:
        render_history()
    else:
        render_empty_state()


def render_empty_state():
    """Render the empty state when no ad has been generated."""
    st.markdown(
        """<div style="background: var(--panel-bg); padding: 2rem; border-radius: 8px; border: 1px solid var(--border-light);">
<h3 style="font-family: var(--font-display); font-size: 1.5rem; color: var(--accent-primary); margin-bottom: 1rem;">How it works</h3>
<p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;"><strong>1. Upload</strong> your product image and company logo</p>
<p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;"><strong>2. Select</strong> your target platform (Instagram or LinkedIn)</p>
<p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;"><strong>3. Choose</strong> the aspect ratio for your ad</p>
<p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;"><strong>4. Enter</strong> your product name</p>
<p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem;"><strong>5. Generate</strong> your professional ad!</p>

<h3 style="font-family: var(--font-display); font-size: 1.25rem; color: var(--accent-primary); margin-bottom: 1rem;">Our AI Agents</h3>
<ul style="color: var(--text-muted); font-size: 0.9rem; padding-left: 1.5rem;">
<li><strong>Analyzer:</strong> Examines your product image</li>
<li><strong>Prompt Engineer:</strong> Creates optimal generation prompt</li>
<li><strong>Designer:</strong> Generates the ad with your logo</li>
<li><strong>Copywriter:</strong> Writes LinkedIn post (if selected)</li>
</ul>
</div>""",
        unsafe_allow_html=True
    )


def render_pomelli_progress():
    """Render pomelli mode: use Gemini API to place black logo on the original photo."""
    st.markdown("---")
    
    # Pomelli-specific prompt: keep the photo as-is, just place the logo
    pomelli_prompt = """
POMELLI MODE - LOGO PLACEMENT ONLY:

ABSOLUTE RULES - DO NOT BREAK ANY OF THESE:
1. DO NOT modify, alter, or change the original photo in ANY way
2. DO NOT change colors, lighting, background, or any visual element of the photo
3. DO NOT add any text, effects, filters, or overlays other than the logo
4. The ONLY change allowed is placing the logo

LOGO RULES - CRITICAL:
- USE THE EXACT LOGO IMAGE PROVIDED - do NOT redraw, recreate, or modify the logo shape/design in ANY way
- The logo must be a PIXEL-PERFECT copy of the provided logo image
- DO NOT change the logo's shape, proportions, text, icons, or any detail whatsoever
- DO NOT simplify, stylize, or reinterpret the logo - use it EXACTLY as given
- The ONLY modification allowed is making the logo BLACK/MONOCHROME in color
- Convert all non-transparent parts of the logo to solid BLACK color

*** MOST IMPORTANT RULE - TRANSPARENT BACKGROUND ***
- The logo MUST NOT have any white box, white rectangle, or white background behind it
- REMOVE all white/light background from the logo completely
- The logo must appear as if it is DIRECTLY ON TOP of the photo with NO background shape behind it
- There should be ZERO visible background, border, box, or container around the logo
- The black logo graphic should sit directly on the photo surface as if printed/stamped on it
- If the original logo has a white/light background, STRIP IT AWAY so only the logo shape remains
- Think of it like a WATERMARK - just the logo shape, nothing else

PLACEMENT INSTRUCTIONS:
- Place the black version of the EXACT provided logo in the TOP-RIGHT CORNER of the image
- Make the logo SMALL - about 5-8% of the image width
- The logo should look professionally placed, subtle and elegant
- Keep the original photo COMPLETELY UNCHANGED underneath

OUTPUT: Return the EXACT same photo with ONLY the exact provided logo (made black, with NO white background) placed in the top-right corner. The logo must blend directly onto the photo like a watermark with NO box or background behind it.
"""
    
    with st.status("Applying logo via AI...", expanded=True) as status:
        try:
            st.write("**Step 1/1:** Using Gemini to place black logo on your image...")
            st.caption("This may take 15-30 seconds...")
            
            # Create a minimal workflow state for Agent 3
            state = create_initial_state(
                platform="instagram",  # doesn't matter for pomelli
                aspect_ratio="1:1",  # will be overridden by actual image size
                product_image=st.session_state.product_image,
                logo_image=st.session_state.logo_image,
                product_name="pomelli",
            )
            # Set the pomelli-specific prompt (skipping agents 1 & 2)
            state["image_generation_prompt"] = pomelli_prompt
            
            # Call Agent 3 directly
            result = agent3_ad_generator_sync(state)
            state = {**state, **result}
            
            if state.get("error"):
                raise Exception(f"Logo placement failed: {state['error']}")
            
            st.write("Logo applied successfully!")
            status.update(label="Done!", state="complete", expanded=False)
            
            # Store result
            st.session_state.workflow_state = state
            if state.get("generated_ad_image"):
                st.session_state.generated_ad = Image.open(io.BytesIO(state["generated_ad_image"]))
            st.session_state.generation_in_progress = False
            st.rerun()
            
        except Exception as e:
            status.update(label="Failed", state="error", expanded=True)
            st.error(f"Error: {str(e)}")
            logger.exception("Pomelli logo placement failed")
            st.session_state.generation_in_progress = False
            
            if st.button("Try Again", use_container_width=True):
                st.session_state.generation_in_progress = True
                st.rerun()


def render_rakhi_progress():
    """Render rakhi mode: generate a photoshoot-level image from a raw rakhi photo."""
    st.markdown("---")
    
    # Load the rakhi prompt from file
    import os
    rakhi_prompt_path = os.path.join(os.path.dirname(__file__), "rakhi-prompt.md")
    try:
        with open(rakhi_prompt_path, "r", encoding="utf-8") as f:
            rakhi_prompt = f.read()
    except FileNotFoundError:
        st.error("rakhi-prompt.md not found! Please ensure the file exists in the project root.")
        st.session_state.generation_in_progress = False
        return
    
    # --- Add variety to each generation ---
    import random
    
    # ONLY light-toned surfaces
    surfaces = [
        "a textured, natural fiber handmade paper with a light, speckled, warm cream-beige color",
        "a smooth, creamy white marble slab with subtle grey veining",
        "a light sandy linen fabric laid flat with a soft, natural weave texture",
        "a pale ivory handmade cotton rag paper with torn edges and visible fibers",
        "a light warm grey concrete surface with a smooth matte finish",
        "a bleached light oak wood surface with subtle pale grain patterns",
        "a soft off-white muslin cloth draped flat with gentle creases",
        "a light pistachio-tinted handmade paper with a delicate speckled texture",
    ]
    
    # Minimal accent combos - only 2 items each, placed at edges
    accent_combos = [
        "a corner of a fresh dark green banana leaf peeking in from the top-left, and a small brass bowl of red kumkum partially visible at the edge",
        "a few sparse rice grains scattered near the top-right corner, and the tip of a banana leaf entering from the left edge",
        "a small brass kumkum bowl at one corner, and 3-4 loose flower petals near the opposite edge",
        "a sprig of fresh green leaves at one corner, and a tiny pinch of scattered turmeric powder at the opposite edge",
        "a single fresh banana leaf corner entering from the left, and a few scattered saffron threads near the bottom edge",
        "a small polished brass diya at one corner, and a sparse trail of rice grains along one edge",
        "a folded edge of gold mesh textile peeking in from one corner, and a tiny brass bowl of kumkum at the opposite corner",
        "a small cluster of 3-4 rice grains at one corner, and a single green betel leaf partially visible at another edge",
    ]
    
    # Just a few petals scattered - very sparse
    petal_styles = [
        "2-3 deep crimson rose petals",
        "3-4 mixed red and yellow rose petals",
        "2-3 soft pink petals",
        "a couple of bright yellow marigold petals",
        "2-3 pale white jasmine buds",
        "3-4 dried rose petals in muted tones",
    ]
    
    # Only bright, airy lighting
    lighting_moods = [
        "soft, diffused natural daylight - bright and airy",
        "clean morning light with a fresh, crisp feel",
        "warm but bright natural window light",
        "soft overcast daylight - even and shadow-free",
        "gentle warm-toned daylight with minimal shadows",
    ]
    
    chosen_surface = random.choice(surfaces)
    chosen_accents = random.choice(accent_combos)
    chosen_petals = random.choice(petal_styles)
    chosen_lighting = random.choice(lighting_moods)
    
    variation_block = f"""

--- VARIATION FOR THIS GENERATION ---
Surface: {chosen_surface}
Edge accents (only these 2 items, partially cropped at edges): {chosen_accents}
Sparse petals (scatter loosely, do NOT overdo): {chosen_petals}
Lighting: {chosen_lighting}

REMINDER: Keep the surface LIGHT. Use ONLY these 2 accent elements at the far edges. Leave lots of empty space. The rakhi must dominate the frame. Do NOT add extra props, figurines, garlands, or objects beyond what is listed above.
--- END VARIATION ---
"""
    # Add logo integration instructions to the rakhi prompt
    logo_instructions = """

--- LOGO INTEGRATION ---
I am also providing a company logo image along with the rakhi image.
LOGO RULES - CRITICAL:
- USE THE EXACT LOGO IMAGE PROVIDED - do NOT redraw, recreate, or modify the logo shape/design in ANY way
- The logo must be a PIXEL-PERFECT copy of the provided logo image
- DO NOT change the logo's shape, proportions, text, icons, or any detail whatsoever
- DO NOT simplify, stylize, or reinterpret the logo - use it EXACTLY as given
- The ONLY modification allowed is making the logo BLACK/MONOCHROME in color
- Convert all non-transparent parts of the logo to solid BLACK color

*** MOST IMPORTANT RULE - TRANSPARENT BACKGROUND ***
- The logo MUST NOT have any white box, white rectangle, or white background behind it
- REMOVE all white/light background from the logo completely
- The logo must appear as if it is DIRECTLY ON TOP of the photo with NO background shape behind it
- There should be ZERO visible background, border, box, or container around the logo
- The black logo graphic should sit directly on the photo surface as if printed/stamped on it

PLACEMENT INSTRUCTIONS:
- Place the black version of the EXACT provided logo in the TOP-RIGHT CORNER of the image
- Make the logo SMALL - about 5-8% of the image width
- The logo should look professionally placed, subtle and elegant
--- END LOGO INTEGRATION ---
"""
    rakhi_prompt = rakhi_prompt + variation_block + logo_instructions
    
    with st.status("Creating rakhi photoshoot...", expanded=True) as status:
        try:
            st.write("**Step 1/1:** Generating photoshoot-level rakhi image with logo...")
            st.caption("This may take 30-60 seconds...")
            
            # Create a minimal workflow state for Agent 3
            state = create_initial_state(
                platform="instagram",
                aspect_ratio="1:1",
                product_image=st.session_state.product_image,
                logo_image=st.session_state.logo_image,
                product_name="rakhi",
            )
            # Set the rakhi-specific prompt (skipping agents 1 & 2)
            state["image_generation_prompt"] = rakhi_prompt
            
            # Call Agent 3 directly
            result = agent3_ad_generator_sync(state)
            state = {**state, **result}
            
            if state.get("error"):
                raise Exception(f"Rakhi photoshoot generation failed: {state['error']}")
            
            st.write("✨ Photoshoot image created successfully!")
            status.update(label="Done!", state="complete", expanded=False)
            
            # Store result
            st.session_state.workflow_state = state
            if state.get("generated_ad_image"):
                st.session_state.generated_ad = Image.open(io.BytesIO(state["generated_ad_image"]))
            st.session_state.generation_in_progress = False
            st.rerun()
            
        except Exception as e:
            status.update(label="Failed", state="error", expanded=True)
            st.error(f"Error: {str(e)}")
            logger.exception("Rakhi photoshoot generation failed")
            st.session_state.generation_in_progress = False
            
            if st.button("Try Again", use_container_width=True):
                st.session_state.generation_in_progress = True
                st.rerun()


def _build_rakhi_prompt():
    """Build the rakhi prompt with random variation and logo instructions. Returns the full prompt string."""
    import random
    
    rakhi_prompt_path = os.path.join(os.path.dirname(__file__), "rakhi-prompt.md")
    with open(rakhi_prompt_path, "r", encoding="utf-8") as f:
        rakhi_prompt = f.read()
    
    # ONLY light-toned surfaces
    surfaces = [
        "a textured, natural fiber handmade paper with a light, speckled, warm cream-beige color",
        "a smooth, creamy white marble slab with subtle grey veining",
        "a light sandy linen fabric laid flat with a soft, natural weave texture",
        "a pale ivory handmade cotton rag paper with torn edges and visible fibers",
        "a light warm grey concrete surface with a smooth matte finish",
        "a bleached light oak wood surface with subtle pale grain patterns",
        "a soft off-white muslin cloth draped flat with gentle creases",
        "a light pistachio-tinted handmade paper with a delicate speckled texture",
    ]
    
    accent_combos = [
        "a corner of a fresh dark green banana leaf peeking in from the top-left, and a small brass bowl of red kumkum partially visible at the edge",
        "a few sparse rice grains scattered near the top-right corner, and the tip of a banana leaf entering from the left edge",
        "a small brass kumkum bowl at one corner, and 3-4 loose flower petals near the opposite edge",
        "a sprig of fresh green leaves at one corner, and a tiny pinch of scattered turmeric powder at the opposite edge",
        "a single fresh banana leaf corner entering from the left, and a few scattered saffron threads near the bottom edge",
        "a small polished brass diya at one corner, and a sparse trail of rice grains along one edge",
        "a folded edge of gold mesh textile peeking in from one corner, and a tiny brass bowl of kumkum at the opposite corner",
        "a small cluster of 3-4 rice grains at one corner, and a single green betel leaf partially visible at another edge",
    ]
    
    petal_styles = [
        "2-3 deep crimson rose petals",
        "3-4 mixed red and yellow rose petals",
        "2-3 soft pink petals",
        "a couple of bright yellow marigold petals",
        "2-3 pale white jasmine buds",
        "3-4 dried rose petals in muted tones",
    ]
    
    lighting_moods = [
        "soft, diffused natural daylight - bright and airy",
        "clean morning light with a fresh, crisp feel",
        "warm but bright natural window light",
        "soft overcast daylight - even and shadow-free",
        "gentle warm-toned daylight with minimal shadows",
    ]
    
    chosen_surface = random.choice(surfaces)
    chosen_accents = random.choice(accent_combos)
    chosen_petals = random.choice(petal_styles)
    chosen_lighting = random.choice(lighting_moods)
    
    variation_block = f"""

--- VARIATION FOR THIS GENERATION ---
Surface: {chosen_surface}
Edge accents (only these 2 items, partially cropped at edges): {chosen_accents}
Sparse petals (scatter loosely, do NOT overdo): {chosen_petals}
Lighting: {chosen_lighting}

REMINDER: Keep the surface LIGHT. Use ONLY these 2 accent elements at the far edges. Leave lots of empty space. The rakhi must dominate the frame. Do NOT add extra props, figurines, garlands, or objects beyond what is listed above.
--- END VARIATION ---
"""
    logo_instructions = """

--- LOGO INTEGRATION ---
I am also providing a company logo image along with the rakhi image.
LOGO RULES - CRITICAL:
- USE THE EXACT LOGO IMAGE PROVIDED - do NOT redraw, recreate, or modify the logo shape/design in ANY way
- The logo must be a PIXEL-PERFECT copy of the provided logo image
- DO NOT change the logo's shape, proportions, text, icons, or any detail whatsoever
- DO NOT simplify, stylize, or reinterpret the logo - use it EXACTLY as given
- The ONLY modification allowed is making the logo BLACK/MONOCHROME in color
- Convert all non-transparent parts of the logo to solid BLACK color

*** MOST IMPORTANT RULE - TRANSPARENT BACKGROUND ***
- The logo MUST NOT have any white box, white rectangle, or white background behind it
- REMOVE all white/light background from the logo completely
- The logo must appear as if it is DIRECTLY ON TOP of the photo with NO background shape behind it
- There should be ZERO visible background, border, box, or container around the logo
- The black logo graphic should sit directly on the photo surface as if printed/stamped on it

PLACEMENT INSTRUCTIONS:
- Place the black version of the EXACT provided logo in the TOP-RIGHT CORNER of the image
- Make the logo SMALL - about 5-8% of the image width
- The logo should look professionally placed, subtle and elegant
--- END LOGO INTEGRATION ---
"""
    return rakhi_prompt + variation_block + logo_instructions


def render_bulk_rakhi_progress():
    """Render bulk rakhi mode: generate photoshoot-level images for multiple rakhi photos."""
    st.markdown("---")
    
    bulk_images = st.session_state.bulk_rakhi_images
    total = len(bulk_images)
    
    st.subheader(f"🪷 Bulk Rakhi Photoshoot — {total} image{'s' if total != 1 else ''}")
    
    # Progress bar and status text
    progress_bar = st.progress(0, text=f"Processing: 0/{total}")
    status_container = st.empty()
    
    # Container for results as they appear
    results_area = st.container()
    
    results = []
    errors = []
    
    try:
        rakhi_prompt_path = os.path.join(os.path.dirname(__file__), "rakhi-prompt.md")
        if not os.path.exists(rakhi_prompt_path):
            st.error("rakhi-prompt.md not found! Please ensure the file exists in the project root.")
            st.session_state.generation_in_progress = False
            st.session_state.bulk_mode_active = False
            return
    except Exception:
        st.error("Could not locate rakhi-prompt.md")
        st.session_state.generation_in_progress = False
        st.session_state.bulk_mode_active = False
        return
    
    for idx, (img_bytes, img_pil) in enumerate(bulk_images):
        current = idx + 1
        progress_fraction = idx / total
        progress_bar.progress(progress_fraction, text=f"Processing: {idx}/{total} done — generating image {current}...")
        status_container.info(f"🎨 Generating photoshoot for Rakhi {current}/{total}...")
        
        try:
            # Build a fresh prompt with random variation for each image
            full_prompt = _build_rakhi_prompt()
            
            # Create workflow state for this image
            state = create_initial_state(
                platform="instagram",
                aspect_ratio="1:1",
                product_image=img_bytes,
                logo_image=st.session_state.logo_image,
                product_name="rakhi",
            )
            state["image_generation_prompt"] = full_prompt
            
            # Call Agent 3
            result = agent3_ad_generator_sync(state)
            state = {**state, **result}
            
            if state.get("error"):
                raise Exception(f"Generation failed: {state['error']}")
            
            if state.get("generated_ad_image"):
                generated_img = Image.open(io.BytesIO(state["generated_ad_image"]))
                results.append(generated_img)
                errors.append(None)
                
                # Show the result immediately in the results area
                with results_area:
                    cols = st.columns([1, 3, 1])
                    with cols[1]:
                        st.image(generated_img, caption=f"✅ Rakhi {current}/{total}", use_container_width=True)
            else:
                results.append(None)
                errors.append("No image generated")
                with results_area:
                    st.warning(f"⚠️ Rakhi {current}/{total}: No image was returned")
        
        except Exception as e:
            results.append(None)
            errors.append(str(e))
            logger.exception(f"Bulk rakhi generation failed for image {current}")
            with results_area:
                st.error(f"❌ Rakhi {current}/{total}: {str(e)}")
        
        # Update progress
        progress_bar.progress(current / total, text=f"Processing: {current}/{total} done")
    
    # All done
    success_count = sum(1 for e in errors if e is None)
    fail_count = total - success_count
    
    status_container.empty()
    progress_bar.progress(1.0, text=f"✅ Complete: {success_count}/{total} generated successfully")
    
    # Store results in session state
    st.session_state.bulk_rakhi_results = results
    st.session_state.bulk_rakhi_errors = errors
    st.session_state.bulk_rakhi_progress = total
    st.session_state.bulk_rakhi_total = total
    st.session_state.generation_in_progress = False
    st.session_state.bulk_mode_active = False
    st.session_state.bulk_mode_done = True
    
    # Show summary and action buttons
    if fail_count > 0:
        st.warning(f"⚠️ {fail_count} image(s) failed. View results below.")
    else:
        st.success(f"🎉 All {total} images generated successfully!")
    
    if st.button("View Final Results", type="primary", use_container_width=True):
        st.rerun()


def render_bulk_rakhi_results():
    """Render the final results grid for bulk rakhi generation."""
    st.markdown("---")
    
    results = st.session_state.bulk_rakhi_results
    errors = st.session_state.bulk_rakhi_errors
    total = st.session_state.bulk_rakhi_total
    
    success_count = sum(1 for e in errors if e is None)
    fail_count = total - success_count
    
    st.subheader("🪷 Bulk Rakhi Photoshoot Results")
    
    # Summary bar
    if fail_count == 0:
        st.success(f"✅ All {total} images generated successfully!")
    else:
        st.warning(f"✅ {success_count}/{total} generated successfully — ❌ {fail_count} failed")
    
    st.divider()
    
    # Image grid — 2 columns
    successful_images = []
    for idx in range(total):
        if results[idx] is not None:
            successful_images.append((idx, results[idx]))
    
    if successful_images:
        cols = st.columns(2)
        for i, (idx, img) in enumerate(successful_images):
            with cols[i % 2]:
                st.image(img, caption=f"Rakhi {idx + 1}", use_container_width=True)
                
                # Individual download button
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                st.download_button(
                    f"⬇️ Download Rakhi {idx + 1}",
                    data=img_buffer.getvalue(),
                    file_name=f"rakhi_{idx + 1}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"download_rakhi_{idx}"
                )
    
    # Show failures
    if fail_count > 0:
        with st.expander(f"❌ {fail_count} Failed Generation(s)", expanded=False):
            for idx in range(total):
                if errors[idx] is not None:
                    st.error(f"Rakhi {idx + 1}: {errors[idx]}")
    
    st.divider()
    
    # Action buttons
    action_cols = st.columns(3)
    
    with action_cols[0]:
        # Download All as ZIP
        if successful_images:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, img in successful_images:
                    img_bytes_buf = io.BytesIO()
                    img.save(img_bytes_buf, format="PNG")
                    zf.writestr(f"rakhi_{idx + 1}.png", img_bytes_buf.getvalue())
            
            st.download_button(
                f"📦 Download All ({len(successful_images)} images)",
                data=zip_buffer.getvalue(),
                file_name="rakhi_photoshoot_bulk.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary"
            )
    
    with action_cols[1]:
        # Regenerate failed
        if fail_count > 0:
            if st.button(f"🔄 Retry {fail_count} Failed", use_container_width=True):
                # Keep only failed images for reprocessing
                failed_images = []
                for idx in range(total):
                    if errors[idx] is not None:
                        failed_images.append(st.session_state.bulk_rakhi_images[idx])
                
                st.session_state.bulk_rakhi_images = failed_images
                st.session_state.bulk_mode_done = False
                st.session_state.bulk_mode_active = True
                st.session_state.generation_in_progress = True
                st.session_state.bulk_rakhi_results = []
                st.session_state.bulk_rakhi_errors = []
                st.session_state.bulk_rakhi_progress = 0
                st.session_state.bulk_rakhi_total = len(failed_images)
                st.rerun()
    
    with action_cols[2]:
        # New batch
        if st.button("✨ New Batch", use_container_width=True):
            st.session_state.bulk_mode_done = False
            st.session_state.bulk_mode_active = False
            st.session_state.bulk_rakhi_images = []
            st.session_state.bulk_rakhi_results = []
            st.session_state.bulk_rakhi_errors = []
            st.session_state.bulk_rakhi_progress = 0
            st.session_state.bulk_rakhi_total = 0
            st.session_state.generated_ad = None
            st.session_state.product_image = None
            st.session_state.logo_image = None
            st.rerun()


def render_generation_progress():
    """Render the generation progress indicator and run workflow with real-time updates."""
    st.markdown("---")
    
    # Determine total steps
    is_linkedin = st.session_state.platform == "linkedin"
    total_steps = 4 if is_linkedin else 3
    
    # Create initial state
    state = create_initial_state(
        platform=st.session_state.platform,
        aspect_ratio=st.session_state.aspect_ratio,
        product_image=st.session_state.product_image,
        logo_image=st.session_state.logo_image,
        product_name=st.session_state.product_name,
        selected_category=st.session_state.selected_category,
        additional_comments=st.session_state.additional_comments.strip() if st.session_state.additional_comments else None,
    )
    
    logger.info(f"Starting workflow {state['workflow_id']}")
    logger.info(f"[DEBUG] selected_category passed to workflow: '{state.get('selected_category')}'")
    
    # Use st.status for real-time progress updates
    with st.status("Generating your ad...", expanded=True) as status:
        try:
            # Step 1: Product Analyzer
            st.write("**Step 1/{}:** Analyzing your product...".format(total_steps))
            result = agent1_product_analyzer_sync(state)
            state = {**state, **result}  # Merge result into state
            if state.get("error"):
                raise Exception(f"Product analysis failed: {state['error']}")
            st.write("Product analyzed successfully!")
            
            # Step 2: Prompt Generator
            st.write("**Step 2/{}:** Creating generation prompt...".format(total_steps))
            result = agent2_prompt_generator_sync(state)
            state = {**state, **result}  # Merge result into state
            if state.get("error"):
                raise Exception(f"Prompt generation failed: {state['error']}")
            st.write("Prompt created successfully!")
            
            # Step 3: Ad Generator
            st.write("**Step 3/{}:** Generating your ad image...".format(total_steps))
            st.caption("This may take 30-60 seconds...")
            result = agent3_ad_generator_sync(state)
            state = {**state, **result}  # Merge result into state
            if state.get("error"):
                raise Exception(f"Ad generation failed: {state['error']}")
            st.write("Ad generated successfully!")
            
            # Step 4: LinkedIn Text (if applicable)
            if is_linkedin:
                st.write("**Step 4/{}:** Writing LinkedIn post...".format(total_steps))
                result = agent4_linkedin_text_sync(state)
                state = {**state, **result}  # Merge result into state
                if state.get("error"):
                    raise Exception(f"LinkedIn text generation failed: {state['error']}")
                st.write("LinkedIn post written successfully!")
            
            # Success!
            status.update(label="Generation complete!", state="complete", expanded=False)
            
            # Store results in session state
            st.session_state.workflow_state = state
            if state.get("generated_ad_image"):
                st.session_state.generated_ad = Image.open(io.BytesIO(state["generated_ad_image"]))
            st.session_state.linkedin_text = state.get("linkedin_post_text")
            

            
            logger.info(f"Workflow {state['workflow_id']} completed successfully")
            st.session_state.generation_in_progress = False
            st.rerun()
            
        except Exception as e:
            status.update(label="Generation failed", state="error", expanded=True)
            st.error(f"Error: {str(e)}")
            logger.exception("Workflow execution failed")
            
            st.session_state.error_message = str(e)
            st.session_state.generation_in_progress = False
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Try Again", use_container_width=True):
                    st.session_state.generation_in_progress = True
                    st.session_state.error_message = None
                    st.rerun()
            with col2:
                if st.button("Start Over", use_container_width=True):
                    st.session_state.generated_ad = None
                    st.session_state.workflow_state = None
                    st.session_state.error_message = None
                    st.rerun()


def render_results():
    """Render the generated ad and options."""
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generated Ad Image
        st.subheader("Generated Ad")
        st.image(st.session_state.generated_ad, use_container_width=True)
        
        # LinkedIn text (if applicable)
        if st.session_state.platform == "linkedin" and st.session_state.linkedin_text:
            st.subheader("LinkedIn Post Copy")
            st.text_area(
                "Copy this text for your LinkedIn post:",
                value=st.session_state.linkedin_text,
                height=250,
                key="linkedin_text_display"
            )
            
            # Copy button hint
            st.caption("Select all and copy the text above for your LinkedIn post.")
    
    with col2:
        st.subheader("Actions")
        
        # Download button
        if st.session_state.generated_ad:
            img_buffer = io.BytesIO()
            st.session_state.generated_ad.save(img_buffer, format="PNG")
            
            product_name_safe = st.session_state.product_name.replace(' ', '_') if st.session_state.product_name else st.session_state.platform
            file_name = f"{product_name_safe}_{st.session_state.platform}_{st.session_state.aspect_ratio.replace(':', 'x')}.png"
            
            st.download_button(
                "Download Ad",
                data=img_buffer.getvalue(),
                file_name=file_name,
                mime="image/png",
                use_container_width=True,
                type="primary"
            )
        
        st.divider()
        
        # Regenerate button
        if st.button("Regenerate", use_container_width=True, help="Generate a new variation with the same settings"):
            st.session_state.generation_in_progress = True
            st.session_state.generated_ad = None
            st.rerun()
        
        # New Creative button
        if st.button("New Creative", use_container_width=True, help="Start fresh with new images"):
            st.session_state.generated_ad = None
            st.session_state.linkedin_text = None
            st.session_state.workflow_state = None
            st.session_state.product_image = None
            st.session_state.logo_image = None
            st.session_state.product_name = ""
            st.session_state.selected_category = "others"
            st.session_state.additional_comments = ""
            st.rerun()
        
        st.divider()
        
        # Show workflow details
        if st.session_state.workflow_state:
            with st.expander("Workflow Details"):
                state = st.session_state.workflow_state
                
                st.markdown(f"**Workflow ID:** `{state.get('workflow_id', 'N/A')[:8]}...`")
                st.markdown(f"**Platform:** {state.get('platform', 'N/A').title()}")
                st.markdown(f"**Aspect Ratio:** {state.get('aspect_ratio', 'N/A')}")
                st.markdown(f"**Template:** {state.get('background_template_used', 'N/A')}")
                
                if state.get("product_analysis"):
                    analysis = state["product_analysis"]
                    st.markdown(f"**Product Type:** {analysis.get('product_type', 'N/A')}")
                    st.markdown(f"**Category:** {analysis.get('product_category', 'N/A')}")


def render_history():
    """Render the workflow history view."""
    st.markdown("---")
    st.subheader("Generation History")
    
    st.info("History feature is not available yet. Stay tuned!")
    
    # Back button
    if st.button("Back to Generator"):
        st.session_state.show_history = False
        st.rerun()


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@300;400;500;600;700&display=swap');

            :root {
                --bg-base: #0a0a0a;
                --panel-bg: rgba(255, 255, 255, 0.02);
                --border-light: rgba(255, 255, 255, 0.08);
                --accent-primary: #ffffff;
                --accent-glow: rgba(255, 255, 255, 0.1);
                --text-main: #f5f5f5;
                --text-muted: #888888;
                --font-display: 'Instrument Serif', serif;
                --font-body: 'Manrope', sans-serif;
            }
            


            /* Base resets and overlays */
            .stApp {
                background-color: var(--bg-base);
                color: var(--text-main);
                font-family: var(--font-body);
            }

            /* Background gradient blobs & Noise */
            .stApp::before {
                content: "";
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                background: 
                    radial-gradient(circle at 80% 0%, rgba(109, 40, 217, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 0% 100%, rgba(204, 255, 0, 0.08) 0%, transparent 40%);
                pointer-events: none;
                z-index: 0;
            }
            .stApp::after {
                content: "";
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
                opacity: 0.04;
                pointer-events: none;
                z-index: 999;
            }

            /* Typography Overrides */
            h1, h2, h3, h4, h5, h6 {
                font-family: var(--font-display) !important;
                font-weight: 800 !important;
                letter-spacing: -0.03em !important;
                color: var(--text-main) !important;
            }

            /* Hide default streamlit header */
            header[data-testid="stHeader"] {
                background: transparent !important;
            }

            /* Sidebar styling */
            section[data-testid="stSidebar"] {
                background-color: rgba(5, 5, 5, 0.8) !important;
                backdrop-filter: blur(20px);
                border-right: 1px solid var(--border-light);
            }
            
            section[data-testid="stSidebar"] hr {
                border-color: rgba(255,255,255,0.05) !important;
                margin: 1.5rem 0 !important;
            }

            /* Input Fields */
            .stTextInput input, .stTextArea textarea, 
            div[data-testid="stSelectbox"] > div[role="combobox"] {
                background: rgba(255, 255, 255, 0.03) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: var(--text-main) !important;
                border-radius: 8px !important;
                font-family: var(--font-body) !important;
                padding: 0.75rem 1rem !important;
                transition: all 0.3s ease;
            }

            .stTextInput input:focus, .stTextArea textarea:focus, 
            div[data-testid="stSelectbox"] > div[role="combobox"]:focus {
                border-color: var(--accent-primary) !important;
                box-shadow: 0 0 0 1px var(--accent-primary), 0 0 15px var(--accent-glow) !important;
                background: rgba(255, 255, 255, 0.05) !important;
                color: #fff !important;
            }

            /* Form labels and small text */
            label, .st-emotion-cache-1j0zowj, p {
                color: var(--text-muted) !important;
            }
            label {
                font-family: var(--font-display) !important;
                text-transform: uppercase;
                font-size: 0.75rem !important;
                letter-spacing: 0.05em;
                color: var(--text-muted) !important;
                margin-bottom: 0.25rem !important;
            }

            /* Primary Button */
            .stButton > button {
                background: var(--accent-primary) !important;
                color: #000 !important;
                border: none !important;
                font-family: var(--font-display) !important;
                font-weight: 700 !important;
                font-size: 1rem !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                padding: 1rem 2rem !important;
                border-radius: 4px !important;
                transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1) !important;
                box-shadow: 0 4px 15px var(--accent-glow) !important;
                width: 100%;
            }

            .stButton > button:hover {
                transform: translateY(-2px) !important;
                background: #cccccc !important;
                box-shadow: 0 8px 25px rgba(255, 255, 255, 0.2) !important;
            }

            .stButton > button:active {
                transform: translateY(1px) !important;
            }

            /* Overriding typical streamlit "secondary" button look if they aren't primary */
            button[kind="secondary"] {
                background: transparent !important;
                color: var(--text-main) !important;
                border: 1px solid var(--border-light) !important;
                box-shadow: none !important;
            }
            button[kind="secondary"]:hover {
                background: rgba(255, 255, 255, 0.05) !important;
                border-color: rgba(255, 255, 255, 0.2) !important;
                color: #fff !important;
            }

            /* File Uploader Target */
            [data-testid="stFileUploader"] > section {
                background: rgba(255, 255, 255, 0.02) !important;
                border: 1px dashed var(--border-light) !important;
                border-radius: 8px !important;
                transition: all 0.3s ease;
                padding: 2rem !important;
            }
            [data-testid="stFileUploader"] > section:hover {
                border-color: var(--accent-primary) !important;
                background: rgba(204, 255, 0, 0.03) !important;
            }
            [data-testid="stFileUploader"] small {
                font-family: var(--font-body);
                color: var(--text-muted) !important;
            }

            /* Selectbox text dropdown styling */
            div[data-baseweb="select"] li {
                background: #000;
                color: #fff;
            }

            /* Radio buttons container */
            [role="radiogroup"] {
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            .stRadio [role="radio"] {
                background-color: rgba(255, 255, 255, 0.02);
                padding: 0.4rem 0.8rem;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
            }
            .stRadio [role="radio"][aria-checked="true"] {
                background-color: rgba(255, 255, 255, 0.1);
                border-color: var(--accent-primary);
                color: var(--text-main);
            }
            .stRadio [role="radio"] div {
                color: inherit !important;
            }
            
            /* Hide the default radio circle marker to look like pill buttons */
            .stRadio [role="radio"] div[data-testid="stMarkdownContainer"] p {
                color: inherit !important;
            }
            .stRadio [role="radio"] > div:first-child {
                display: none;
            }

            /* Expander/Accordions */
            [data-testid="stExpander"] {
                background: var(--panel-bg) !important;
                border: 1px solid var(--border-light) !important;
                border-radius: 8px !important;
                backdrop-filter: blur(10px);
            }
            [data-testid="stExpander"] summary {
                font-family: var(--font-display);
                font-size: 1.1rem;
                color: var(--accent-primary);
            }

            /* Progress and Status Bars */
            .stProgress > div > div > div {
                background-color: var(--accent-primary) !important;
            }
            [data-testid="stStatusWidget"] {
                background: var(--panel-bg) !important;
                border: 1px solid var(--border-light) !important;
                border-radius: 8px !important;
            }
            
            /* Status text color */
            [data-testid="stStatusWidget"] label {
                color: var(--text-main) !important;
            }

            /* Markdown content text colors */
            .stMarkdown p {
                color: var(--text-muted);
            }
            .stMarkdown strong {
                color: var(--text-main);
            }

            /* Custom Selection */
            ::selection {
                background: var(--accent-primary);
                color: #000;
            }

            /* Animation classes */
            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-20px); filter: blur(5px); }
                to { opacity: 1; transform: translateY(0); filter: blur(0); }
            }
            .animated-hero {
                animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Render UI components
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
