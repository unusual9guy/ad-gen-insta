"""Workflow state schema for the Ad Generator LangGraph workflow."""

from typing import TypedDict, Optional, Literal, Any
from dataclasses import dataclass, field
import uuid
from datetime import datetime


class ProductAnalysis(TypedDict):
    """Structured output from Agent 1 - Product Analyzer."""
    product_type: str
    product_category: str
    current_angle: str
    visual_characteristics: dict[str, str]
    colors: list[str]
    materials: list[str]
    positioning_recommendation: str
    composition_notes: str
    needs_angle_regeneration: bool
    angle_regeneration_reason: Optional[str]


class WorkflowState(TypedDict):
    """
    Complete state for the Ad Generator workflow.
    
    This state is passed between all agents in the LangGraph workflow
    and contains all inputs, intermediate results, and final outputs.
    """
    
    # ===== User Inputs =====
    platform: Literal["instagram", "linkedin"]
    aspect_ratio: str
    product_image: bytes
    logo_image: bytes
    product_name: str
    selected_category: str  # User-selected category from templates (or "others" for auto-detect)
    additional_comments: Optional[str]  # Extra instructions to append to prompt
    
    # ===== Agent 1 Outputs: Product Analyzer (Gemini) =====
    product_analysis: Optional[ProductAnalysis]
    
    # ===== Agent 2 Outputs: Prompt Generator (Claude) =====
    image_generation_prompt: Optional[str]
    background_template_used: Optional[str]
    
    # ===== Agent 3 Outputs: Ad Generator (Imagen 3) =====
    raw_generated_image: Optional[bytes]
    generated_ad_image: Optional[bytes]  # After logo overlay
    
    # ===== Agent 4 Outputs: LinkedIn Text (Gemini) =====
    linkedin_post_text: Optional[str]
    
    # ===== Workflow Metadata =====
    workflow_id: str
    created_at: str
    current_agent: Optional[str]
    agent_history: list[dict[str, Any]]
    
    # ===== Error Handling =====
    error: Optional[str]
    error_agent: Optional[str]


def create_initial_state(
    platform: Literal["instagram", "linkedin"],
    aspect_ratio: str,
    product_image: bytes,
    logo_image: bytes,
    product_name: str,
    selected_category: str = "others",
    additional_comments: Optional[str] = None,
) -> WorkflowState:
    """
    Create an initial workflow state with user inputs.
    
    Args:
        platform: Target platform (instagram or linkedin)
        aspect_ratio: Desired aspect ratio for the ad
        product_image: Raw bytes of the product image
        logo_image: Raw bytes of the company logo
        product_name: Name of the product
        selected_category: User-selected category from templates (or "others" for auto-detect)
        additional_comments: Extra instructions to append to the prompt
        
    Returns:
        Initialized WorkflowState ready for workflow execution
    """
    return WorkflowState(
        # User inputs
        platform=platform,
        aspect_ratio=aspect_ratio,
        product_image=product_image,
        logo_image=logo_image,
        product_name=product_name,
        selected_category=selected_category,
        additional_comments=additional_comments,
        
        # Agent outputs (initially None)
        product_analysis=None,
        image_generation_prompt=None,
        background_template_used=None,
        raw_generated_image=None,
        generated_ad_image=None,
        linkedin_post_text=None,
        
        # Metadata
        workflow_id=str(uuid.uuid4()),
        created_at=datetime.utcnow().isoformat(),
        current_agent=None,
        agent_history=[],
        
        # Error handling
        error=None,
        error_agent=None,
    )
