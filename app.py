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
    if platform == "instagram":
        return settings.instagram_aspect_ratios
    else:
        return settings.linkedin_aspect_ratios


def render_sidebar():
    """Render the sidebar with all input controls."""
    with st.sidebar:
        st.markdown("## Ad Configuration")
        
        # Step 1: Product Image Upload
        st.markdown("### 1. Product Image")
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
        
        # Step 2: Logo Upload
        st.markdown("### 2. Company Logo")
        logo_file = st.file_uploader(
            "Upload your company logo",
            type=["png", "jpg", "jpeg", "webp"],
            key="logo_uploader",
            help="Logo will be converted to monochrome and placed in top-right corner"
        )
        
        if logo_file:
            result = process_uploaded_image(logo_file)
            if result:
                st.session_state.logo_image = result[0]
                st.session_state.logo_image_preview = result[1]
                st.image(result[1], caption="Logo Preview", use_container_width=True)
        
        st.divider()
        
        # Step 3: Platform Selection
        st.markdown("### 3. Platform")
        platform = st.radio(
            "Select target platform",
            options=["instagram", "linkedin"],
            format_func=lambda x: "Instagram" if x == "instagram" else "LinkedIn",
            key="platform_selector",
            horizontal=True
        )
        st.session_state.platform = platform
        
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
        
        # Generate Button
        can_generate = all([
            st.session_state.product_image,
            st.session_state.logo_image,
            st.session_state.product_name.strip()
        ])
        
        generate_button = st.button(
            "Generate Ad",
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
                missing.append("product image")
            if not st.session_state.logo_image:
                missing.append("company logo")
            if not st.session_state.product_name.strip():
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
            
            file_name = f"{st.session_state.product_name.replace(' ', '_')}_{st.session_state.platform}_{st.session_state.aspect_ratio.replace(':', 'x')}.png"
            
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
