"""Vector store service using ChromaDB for workflow history and prompt storage."""

import chromadb
from chromadb.config import Settings as ChromaSettings
import json
from typing import Optional, Any
from datetime import datetime
import hashlib
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for the Ad Generator.
    
    Manages three collections:
    1. workflow_history - Complete workflow execution records
    2. generated_prompts - Image generation prompts for similarity search
    3. user_preferences - User settings and patterns
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_path: Path for persistent storage. Uses settings default if None.
        """
        self.persist_path = persist_path or settings.chromadb_path
        
        # Ensure directory exists
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Initialize collections
        self._init_collections()
        
        logger.info(f"VectorStore initialized at {self.persist_path}")
    
    def _init_collections(self):
        """Initialize or get the required collections."""
        # Workflow history collection
        self.workflows = self.client.get_or_create_collection(
            name="workflow_history",
            metadata={"description": "Complete workflow execution records"}
        )
        
        # Generated prompts collection with embedding function
        self.prompts = self.client.get_or_create_collection(
            name="generated_prompts",
            metadata={"description": "Image generation prompts for similarity search"}
        )
        
        # User preferences collection
        self.preferences = self.client.get_or_create_collection(
            name="user_preferences",
            metadata={"description": "User settings and patterns"}
        )
    
    def _generate_id(self, *args) -> str:
        """Generate a unique ID from the given arguments."""
        content = "_".join(str(arg) for arg in args)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    # ==================== Workflow History ====================
    
    def save_workflow(
        self,
        workflow_id: str,
        platform: str,
        aspect_ratio: str,
        product_name: str,
        product_analysis: dict,
        image_generation_prompt: str,
        background_template: str,
        linkedin_text: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> str:
        """
        Save a completed workflow to history.
        
        Args:
            workflow_id: Unique workflow identifier
            platform: Target platform (instagram/linkedin)
            aspect_ratio: Selected aspect ratio
            product_name: Name of the product
            product_analysis: Analysis from Agent 1
            image_generation_prompt: Prompt from Agent 2
            background_template: Template used
            linkedin_text: Generated LinkedIn text (if applicable)
            success: Whether workflow completed successfully
            error: Error message if failed
            
        Returns:
            The workflow ID
        """
        try:
            # Create metadata
            metadata = {
                "workflow_id": workflow_id,
                "platform": platform,
                "aspect_ratio": aspect_ratio,
                "product_name": product_name,
                "product_type": product_analysis.get("product_type", "unknown"),
                "product_category": product_analysis.get("product_category", "unknown"),
                "background_template": background_template,
                "success": success,
                "error": error or "",
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Create document text for embedding
            document = f"""
Product: {product_name}
Type: {product_analysis.get('product_type', 'unknown')}
Category: {product_analysis.get('product_category', 'unknown')}
Colors: {', '.join(product_analysis.get('colors', []))}
Platform: {platform}
Template: {background_template}
Prompt: {image_generation_prompt[:500]}
"""
            
            self.workflows.add(
                ids=[workflow_id],
                documents=[document],
                metadatas=[metadata]
            )
            
            logger.info(f"Saved workflow {workflow_id} to history")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            raise
    
    def get_workflow(self, workflow_id: str) -> Optional[dict]:
        """
        Retrieve a workflow by ID.
        
        Args:
            workflow_id: The workflow ID to retrieve
            
        Returns:
            Workflow data or None if not found
        """
        try:
            result = self.workflows.get(
                ids=[workflow_id],
                include=["documents", "metadatas"]
            )
            
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0] if result["documents"] else None,
                    "metadata": result["metadatas"][0] if result["metadatas"] else None,
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow: {e}")
            return None
    
    def get_recent_workflows(self, limit: int = 10, platform: Optional[str] = None) -> list[dict]:
        """
        Get recent workflows, optionally filtered by platform.
        
        Args:
            limit: Maximum number of workflows to return
            platform: Optional platform filter
            
        Returns:
            List of workflow records
        """
        try:
            where = {"platform": platform} if platform else None
            
            result = self.workflows.get(
                limit=limit,
                where=where,
                include=["documents", "metadatas"]
            )
            
            workflows = []
            for i, wf_id in enumerate(result["ids"]):
                workflows.append({
                    "id": wf_id,
                    "document": result["documents"][i] if result["documents"] else None,
                    "metadata": result["metadatas"][i] if result["metadatas"] else None,
                })
            
            # Sort by created_at descending
            workflows.sort(
                key=lambda x: x["metadata"].get("created_at", "") if x["metadata"] else "",
                reverse=True
            )
            
            return workflows[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent workflows: {e}")
            return []
    
    # ==================== Prompt Storage ====================
    
    def save_prompt(
        self,
        prompt: str,
        product_name: str,
        product_category: str,
        template_used: str,
        platform: str,
        aspect_ratio: str,
    ) -> str:
        """
        Save a generated prompt for similarity search.
        
        Args:
            prompt: The image generation prompt
            product_name: Name of the product
            product_category: Category of the product
            template_used: Background template used
            platform: Target platform
            aspect_ratio: Selected aspect ratio
            
        Returns:
            The prompt ID
        """
        try:
            prompt_id = self._generate_id(prompt, datetime.utcnow().isoformat())
            
            metadata = {
                "product_name": product_name,
                "product_category": product_category,
                "template_used": template_used,
                "platform": platform,
                "aspect_ratio": aspect_ratio,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            self.prompts.add(
                ids=[prompt_id],
                documents=[prompt],
                metadatas=[metadata]
            )
            
            logger.info(f"Saved prompt {prompt_id}")
            return prompt_id
            
        except Exception as e:
            logger.error(f"Failed to save prompt: {e}")
            raise
    
    def find_similar_prompts(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> list[dict]:
        """
        Find prompts similar to the given query.
        
        Args:
            query: Search query (product description or prompt fragment)
            n_results: Number of results to return
            category_filter: Optional category filter
            
        Returns:
            List of similar prompts with metadata
        """
        try:
            where = {"product_category": category_filter} if category_filter else None
            
            results = self.prompts.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            prompts = []
            if results["ids"] and results["ids"][0]:
                for i, prompt_id in enumerate(results["ids"][0]):
                    prompts.append({
                        "id": prompt_id,
                        "prompt": results["documents"][0][i] if results["documents"] else None,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                        "similarity": 1 - results["distances"][0][i] if results["distances"] else 0,
                    })
            
            return prompts
            
        except Exception as e:
            logger.error(f"Failed to find similar prompts: {e}")
            return []
    
    # ==================== User Preferences ====================
    
    def save_preference(self, key: str, value: Any, user_id: str = "default") -> None:
        """
        Save a user preference.
        
        Args:
            key: Preference key
            value: Preference value (will be JSON serialized)
            user_id: User identifier
        """
        try:
            pref_id = f"{user_id}_{key}"
            
            # Check if exists and update, or add new
            existing = self.preferences.get(ids=[pref_id])
            
            document = json.dumps({"key": key, "value": value})
            metadata = {
                "user_id": user_id,
                "key": key,
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            if existing["ids"]:
                self.preferences.update(
                    ids=[pref_id],
                    documents=[document],
                    metadatas=[metadata]
                )
            else:
                self.preferences.add(
                    ids=[pref_id],
                    documents=[document],
                    metadatas=[metadata]
                )
            
            logger.debug(f"Saved preference {key} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save preference: {e}")
    
    def get_preference(self, key: str, user_id: str = "default", default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            user_id: User identifier
            default: Default value if not found
            
        Returns:
            The preference value or default
        """
        try:
            pref_id = f"{user_id}_{key}"
            result = self.preferences.get(ids=[pref_id], include=["documents"])
            
            if result["ids"] and result["documents"]:
                data = json.loads(result["documents"][0])
                return data.get("value", default)
            
            return default
            
        except Exception as e:
            logger.error(f"Failed to get preference: {e}")
            return default
    
    # ==================== Utility Methods ====================
    
    def get_stats(self) -> dict:
        """Get statistics about stored data."""
        return {
            "total_workflows": self.workflows.count(),
            "total_prompts": self.prompts.count(),
            "total_preferences": self.preferences.count(),
        }
    
    def clear_all(self) -> None:
        """Clear all data from all collections. Use with caution."""
        self.client.delete_collection("workflow_history")
        self.client.delete_collection("generated_prompts")
        self.client.delete_collection("user_preferences")
        self._init_collections()
        logger.warning("All vector store data cleared")


# Singleton instance
vector_store = VectorStore()
