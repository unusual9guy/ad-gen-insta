"""
Multi-Agent Product Ad Generator
================================
A Streamlit application that generates professional Instagram and LinkedIn 
product ads using a multi-agent LangGraph workflow.
"""

import streamlit as st
from PIL import Image
import io
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
    """Render the sidebar with all input controls."""
    with st.sidebar:
        st.markdown("## Ad Configuration")
        
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
        
        # Step 2: Product Image Upload (or Rakhi Image for rakhi mode)
        if is_rakhi:
            st.markdown("### 2. Rakhi Image")
            product_file = st.file_uploader(
                "Upload your rakhi photo",
                type=["png", "jpg", "jpeg", "webp"],
                key="product_uploader",
                help="Upload a clear photo of your rakhi for photoshoot-level enhancement"
            )
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
                st.image(result[1], caption="Rakhi Preview" if is_rakhi else "Product Preview", use_container_width=True)
        
        st.divider()
        
        # Step 3: Logo Upload (skip for rakhi mode)
        if not is_rakhi:
            st.markdown("### 3. Company Logo")
            logo_file = st.file_uploader(
                "Upload your company logo",
                type=["png", "jpg", "jpeg", "webp"],
                key="logo_uploader",
                help="Logo will be converted to black and placed in top-right corner" if is_pomelli else "Logo will be converted to monochrome and placed in top-right corner"
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
            st.info("🪷 Rakhi mode: Upload a raw rakhi photo and we'll create a stunning photoshoot-level image with traditional Indian ceremonial props and styling.")
        
        # Generate Button
        if is_rakhi:
            can_generate = bool(st.session_state.product_image)
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
        
        generate_button = st.button(
            "✨ Create Rakhi Photoshoot" if is_rakhi else ("Apply Logo" if is_pomelli else "Generate Ad"),
            type="primary",
            disabled=not can_generate or st.session_state.generation_in_progress,
            use_container_width=True
        )
        
        if generate_button:
            st.session_state.generation_in_progress = True
            st.session_state.current_step = 0
            st.session_state.generated_ad = None
            st.session_state.linkedin_text = None
            st.rerun()
        
        if not can_generate:
            missing = []
            if not st.session_state.product_image:
                missing.append("rakhi image" if is_rakhi else "product image")
            if not is_rakhi and not st.session_state.logo_image:
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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Product Ad Generator")
    with col2:
        if st.session_state.generated_ad:
            st.markdown("")  # Spacing
    
    st.markdown("Generate professional product ads for Instagram and LinkedIn using AI agents.")
    
    # Show generation progress or results
    if st.session_state.generation_in_progress:
        if st.session_state.platform == "rakhi":
            render_rakhi_progress()
        elif st.session_state.platform == "pomelli":
            render_pomelli_progress()
        else:
            render_generation_progress()
    elif st.session_state.generated_ad:
        render_results()
    elif st.session_state.show_history:
        render_history()
    else:
        render_empty_state()


def render_empty_state():
    """Render the empty state when no ad has been generated."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            ### How it works
            
            **1. Upload** your product image and company logo
            
            **2. Select** your target platform (Instagram or LinkedIn)
            
            **3. Choose** the aspect ratio for your ad
            
            **4. Enter** your product name
            
            **5. Generate** your professional ad!
            
            ---
            
            ### Our AI Agents
            
            | Agent | Task |
            |-------|------|
            | **Analyzer** | Examines your product image |
            | **Prompt Engineer** | Creates optimal generation prompt |
            | **Designer** | Generates the ad with your logo |
            | **Copywriter** | Writes LinkedIn post (if selected) |
            """
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
    
    surfaces = [
        "a textured, natural fiber handmade paper with a light, speckled, warm grey-beige color",
        "a rich, dark walnut wood surface with visible natural grain patterns",
        "a smooth, creamy white marble slab with subtle grey veining",
        "a rustic terracotta clay tile with a warm, earthy orange-brown tone",
        "a deep burgundy raw silk fabric draped flat with fine texture visible",
        "an aged brass tray with a beautiful green-gold patina",
        "a woven jute mat with a natural golden-tan color and visible weave texture",
        "a dark slate stone surface with subtle grey-blue tones and matte finish",
    ]
    
    accent_sets = [
        "a fresh dark green banana leaf segment, a polished brass bowl of vibrant red kumkum powder, scattered uncooked basmati rice grains",
        "a cluster of fresh marigold flowers (orange and yellow), a small silver bowl of turmeric paste, a few cardamom pods",
        "fresh jasmine flower strings, a copper kalash (small pot), scattered red and yellow rose petals",
        "a betel leaf arrangement, a small clay diya with a cotton wick, a sprinkle of golden turmeric powder",
        "a mango leaf toran (garland), a brass bell, scattered mogra (jasmine) buds",
        "neem leaves arranged decoratively, a silver plate with sindoor, loose saffron threads",
        "lotus petals (pink and white), a small sandalwood piece, scattered dried marigold petals",
        "tulsi (holy basil) sprigs, a decorative brass diya, a trail of bright orange kumkum",
    ]
    
    textiles = [
        "a delicate, fine-weave gold mesh textile",
        "a piece of rich red Banarasi silk with gold zari border",
        "a soft pastel pink chiffon dupatta with silver thread embroidery",
        "a deep emerald green velvet fabric with gold embroidery accents",
        "a cream-colored raw silk piece with subtle gold brocade",
        "a royal purple satin fabric with intricate gold threadwork",
        "a traditional bandhani (tie-dye) fabric in red and yellow",
        "a sheer gold organza fabric catching light beautifully",
    ]
    
    flower_scatters = [
        "deep crimson and bright yellow rose petals",
        "orange marigold petals and white jasmine buds",
        "pink lotus petals and golden champa flowers",
        "red hibiscus petals and tiny white mogra flowers",
        "purple and white orchid petals",
        "mixed marigold petals in sunset shades (orange, yellow, gold)",
        "pale pink and deep magenta bougainvillea petals",
        "dried rose buds and fresh lavender sprigs",
    ]
    
    lighting_moods = [
        "warm golden-hour sunlight with soft amber tones",
        "cool morning daylight with a fresh, clean feel",
        "rich, warm candlelight-inspired glow with deep shadows",
        "bright midday diffused light with crisp, clear tones",
        "soft dusk lighting with rose-gold warmth",
        "dramatic side-lighting with elegant shadow play",
    ]
    
    bonus_props = [
        "a small decorative mirror and a few whole cloves",
        "a sandalwood incense stick (unlit) and some dried flower buds",
        "a tiny brass Ganesha figurine and some roli grains",
        "a decorative silver coin and a few almonds",
        "a miniature peacock feather and betel nuts",
        "a piece of raw sugarcane and some mishri (rock sugar) crystals",
        "a small conch shell and some dried turmeric roots",
        "a cotton thread spool (red and gold) and some whole black cardamom",
    ]
    
    chosen_surface = random.choice(surfaces)
    chosen_accents = random.choice(accent_sets)
    chosen_textile = random.choice(textiles)
    chosen_flowers = random.choice(flower_scatters)
    chosen_lighting = random.choice(lighting_moods)
    chosen_bonus = random.choice(bonus_props)
    
    variation_block = f"""

--- VARIATION FOR THIS GENERATION (override the default accents with these) ---
Surface: {chosen_surface}
Accent elements: {chosen_accents}
Textile/fabric: {chosen_textile}
Scattered flowers: {chosen_flowers}
Lighting mood: {chosen_lighting}
Bonus props: {chosen_bonus}
Use these specific elements instead of the defaults to create a UNIQUE composition. Keep the same overall structure (top-down flat-lay, rakhi centered, ceremonial feel) but make this version visually distinct.
--- END VARIATION ---
"""
    rakhi_prompt = rakhi_prompt + variation_block
    
    with st.status("Creating rakhi photoshoot...", expanded=True) as status:
        try:
            st.write("**Step 1/1:** Generating photoshoot-level rakhi image...")
            st.caption("This may take 30-60 seconds...")
            
            # For rakhi mode, we create a 1px transparent logo as a dummy
            # since the ad_generator agent requires a logo_image
            from PIL import Image as PILImage
            dummy_logo = PILImage.new("RGBA", (1, 1), (0, 0, 0, 0))
            dummy_logo_buffer = io.BytesIO()
            dummy_logo.save(dummy_logo_buffer, format="PNG")
            dummy_logo_bytes = dummy_logo_buffer.getvalue()
            
            # Create a minimal workflow state for Agent 3
            state = create_initial_state(
                platform="instagram",
                aspect_ratio="1:1",
                product_image=st.session_state.product_image,
                logo_image=dummy_logo_bytes,
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
        .stButton > button {
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
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
