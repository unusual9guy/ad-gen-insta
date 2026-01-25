"""LangGraph node functions for the Ad Generator workflow."""

import asyncio
from typing import Any
from datetime import datetime

from workflow.state import WorkflowState
from agents.product_analyzer import product_analyzer


def add_to_history(state: WorkflowState, agent_name: str, result: dict[str, Any]) -> list:
    """Add an agent execution to the history."""
    history = state.get("agent_history", []).copy()
    history.append({
        "agent": agent_name,
        "timestamp": datetime.utcnow().isoformat(),
        "success": result.get("error") is None,
        "error": result.get("error"),
    })
    return history


async def agent1_product_analyzer(state: WorkflowState) -> dict[str, Any]:
    """
    Node function for Agent 1: Product Analyzer.
    
    Analyzes the product image using Gemini Vision and extracts
    detailed product information for use by subsequent agents.
    """
    result = await product_analyzer.run(state)
    result["agent_history"] = add_to_history(state, "ProductAnalyzer", result)
    return result


async def agent2_prompt_generator(state: WorkflowState) -> dict[str, Any]:
    """
    Node function for Agent 2: Prompt Generator.
    
    Uses Claude to generate an optimized image generation prompt
    based on the product analysis.
    """
    # TODO: Implement in Phase 3
    from agents.prompt_generator import prompt_generator
    result = await prompt_generator.run(state)
    result["agent_history"] = add_to_history(state, "PromptGenerator", result)
    return result


async def agent3_ad_generator(state: WorkflowState) -> dict[str, Any]:
    """
    Node function for Agent 3: Ad Generator.
    
    Uses Imagen 3 to generate the ad image and applies logo overlay.
    """
    # TODO: Implement in Phase 4
    from agents.ad_generator import ad_generator
    result = await ad_generator.run(state)
    result["agent_history"] = add_to_history(state, "AdGenerator", result)
    return result


async def agent4_linkedin_text(state: WorkflowState) -> dict[str, Any]:
    """
    Node function for Agent 4: LinkedIn Text Generator.
    
    Uses Gemini to generate professional LinkedIn post copy.
    Only executed for LinkedIn platform.
    """
    # TODO: Implement in Phase 5
    from agents.linkedin_text import linkedin_text_generator
    result = await linkedin_text_generator.run(state)
    result["agent_history"] = add_to_history(state, "LinkedInTextGenerator", result)
    return result


# Synchronous wrappers for LangGraph compatibility
def agent1_product_analyzer_sync(state: WorkflowState) -> dict[str, Any]:
    """Synchronous wrapper for agent1_product_analyzer."""
    return asyncio.run(agent1_product_analyzer(state))


def agent2_prompt_generator_sync(state: WorkflowState) -> dict[str, Any]:
    """Synchronous wrapper for agent2_prompt_generator."""
    return asyncio.run(agent2_prompt_generator(state))


def agent3_ad_generator_sync(state: WorkflowState) -> dict[str, Any]:
    """Synchronous wrapper for agent3_ad_generator."""
    return asyncio.run(agent3_ad_generator(state))


def agent4_linkedin_text_sync(state: WorkflowState) -> dict[str, Any]:
    """Synchronous wrapper for agent4_linkedin_text."""
    return asyncio.run(agent4_linkedin_text(state))
