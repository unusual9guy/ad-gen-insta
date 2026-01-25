"""Base agent interface for all AI agents in the workflow."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from workflow.state import WorkflowState
import logging

logger = logging.getLogger(__name__)

# Type variable for agent-specific output types
T = TypeVar('T')


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for all AI agents in the workflow.
    
    Each agent is responsible for a specific task in the ad generation
    pipeline and must implement the `process` method.
    """
    
    # Agent metadata
    name: str = "BaseAgent"
    description: str = "Base agent class"
    
    def __init__(self):
        """Initialize the agent."""
        self.logger = logging.getLogger(f"agents.{self.name}")
    
    @abstractmethod
    async def process(self, state: WorkflowState) -> dict[str, Any]:
        """
        Process the current workflow state and return updates.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of state updates to merge into workflow state
        """
        pass
    
    def validate_inputs(self, state: WorkflowState) -> bool:
        """
        Validate that required inputs are present in the state.
        
        Override in subclasses to add specific validation logic.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        return True
    
    def log_start(self, state: WorkflowState):
        """Log agent execution start."""
        self.logger.info(f"Starting {self.name} for workflow {state['workflow_id']}")
    
    def log_complete(self, state: WorkflowState):
        """Log agent execution completion."""
        self.logger.info(f"Completed {self.name} for workflow {state['workflow_id']}")
    
    def log_error(self, state: WorkflowState, error: Exception):
        """Log agent execution error."""
        self.logger.error(
            f"Error in {self.name} for workflow {state['workflow_id']}: {error}"
        )
    
    async def run(self, state: WorkflowState) -> dict[str, Any]:
        """
        Execute the agent with logging and error handling.
        
        This is the main entry point for agent execution in the workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of state updates
        """
        self.log_start(state)
        
        try:
            # Validate inputs
            if not self.validate_inputs(state):
                raise ValueError(f"Invalid inputs for {self.name}")
            
            # Process
            result = await self.process(state)
            
            # Add agent to history
            result["current_agent"] = self.name
            
            self.log_complete(state)
            return result
            
        except Exception as e:
            self.log_error(state, e)
            return {
                "error": str(e),
                "error_agent": self.name,
                "current_agent": self.name,
            }
