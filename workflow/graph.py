"""LangGraph workflow definition for the Ad Generator."""

from langgraph.graph import StateGraph, END
from typing import Literal

from workflow.state import WorkflowState
from workflow.nodes import (
    agent1_product_analyzer_sync,
    agent2_prompt_generator_sync,
    agent3_ad_generator_sync,
    agent4_linkedin_text_sync,
)


def should_continue_after_error(state: WorkflowState) -> Literal["continue", "end"]:
    """Check if workflow should continue or end due to error."""
    if state.get("error"):
        return "end"
    return "continue"


def route_by_platform(state: WorkflowState) -> Literal["linkedin_text", "end"]:
    """
    Route to LinkedIn text generation or end based on platform.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name or END
    """
    if state.get("error"):
        return "end"
    
    if state["platform"] == "linkedin":
        return "linkedin_text"
    return "end"


def create_workflow_graph() -> StateGraph:
    """
    Create and configure the LangGraph workflow for ad generation.
    
    The workflow follows this flow:
    1. Agent 1: Product Analyzer (Gemini) - Analyzes product image
    2. Agent 2: Prompt Generator (Claude) - Creates image gen prompt
    3. Agent 3: Ad Generator (Imagen 3) - Generates ad image
    4. Agent 4: LinkedIn Text (Gemini) - [LinkedIn only] Generates post copy
    
    Returns:
        Configured StateGraph ready for compilation
    """
    # Create the graph with WorkflowState
    workflow = StateGraph(WorkflowState)
    
    # Add nodes for each agent
    workflow.add_node("product_analyzer", agent1_product_analyzer_sync)
    workflow.add_node("prompt_generator", agent2_prompt_generator_sync)
    workflow.add_node("ad_generator", agent3_ad_generator_sync)
    workflow.add_node("linkedin_text", agent4_linkedin_text_sync)
    
    # Define the edges (flow)
    # Start -> Agent 1
    workflow.set_entry_point("product_analyzer")
    
    # Agent 1 -> Agent 2 (with error check)
    workflow.add_conditional_edges(
        "product_analyzer",
        should_continue_after_error,
        {
            "continue": "prompt_generator",
            "end": END,
        }
    )
    
    # Agent 2 -> Agent 3 (with error check)
    workflow.add_conditional_edges(
        "prompt_generator",
        should_continue_after_error,
        {
            "continue": "ad_generator",
            "end": END,
        }
    )
    
    # Agent 3 -> Agent 4 (if LinkedIn) or END (if Instagram)
    workflow.add_conditional_edges(
        "ad_generator",
        route_by_platform,
        {
            "linkedin_text": "linkedin_text",
            "end": END,
        }
    )
    
    # Agent 4 -> END
    workflow.add_edge("linkedin_text", END)
    
    return workflow


def compile_workflow():
    """
    Compile the workflow graph for execution.
    
    Returns:
        Compiled workflow ready for invocation
    """
    graph = create_workflow_graph()
    return graph.compile()


# Pre-compiled workflow instance for import convenience
ad_generator_workflow = compile_workflow()
