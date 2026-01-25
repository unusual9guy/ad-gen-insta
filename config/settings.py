"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys - Separate keys for each agent
    # Agent 1: Product Analyzer (Gemini)
    product_analyzer_api_key: Optional[str] = Field(default=None, alias="PRODUCT_ANALYZER_API_KEY")
    
    # Agent 2: Prompt Generator (Gemini)
    prompt_generator_api_key: Optional[str] = Field(default=None, alias="PROMPT_GENERATOR_API_KEY")
    
    # Agent 3: Ad Generator (Google Image Generation)
    ad_generator_api_key: Optional[str] = Field(default=None, alias="AD_GENERATOR_API_KEY")
    
    # Agent 4: LinkedIn Text Generator (Gemini)
    linkedin_text_api_key: Optional[str] = Field(default=None, alias="LINKEDIN_TEXT_API_KEY")
    
    # Fallback keys (used if agent-specific keys not provided)
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    
    # App Configuration
    app_name: str = "Product Ad Generator"
    app_version: str = "1.0.0"
    
    # Platform Configuration
    instagram_aspect_ratios: list[str] = ["1:1", "4:5"]
    linkedin_aspect_ratios: list[str] = ["1:1", "1.91:1"]
    default_aspect_ratio: str = "1:1"
    
    # Image Processing
    logo_size_percentage: float = 0.06  # 6% of ad width
    logo_padding_percentage: float = 0.03  # 3% padding from edges
    
    # Vector DB
    chromadb_path: str = "./data/chromadb"
    
    # Paths
    assets_path: Path = Path("./assets")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_agent1_key(self) -> Optional[str]:
        """Get API key for Agent 1 (Product Analyzer - Gemini)."""
        return self.product_analyzer_api_key or self.google_api_key
    
    def get_agent2_key(self) -> Optional[str]:
        """Get API key for Agent 2 (Prompt Generator - Gemini)."""
        return self.prompt_generator_api_key or self.google_api_key
    
    def get_agent3_key(self) -> Optional[str]:
        """Get API key for Agent 3 (Ad Generator - Google Image Gen)."""
        return self.ad_generator_api_key or self.google_api_key
    
    def get_agent4_key(self) -> Optional[str]:
        """Get API key for Agent 4 (LinkedIn Text - Gemini)."""
        return self.linkedin_text_api_key or self.google_api_key
    
    @property
    def effective_gemini_key(self) -> Optional[str]:
        """Return the default Gemini/Google API key (deprecated, use agent-specific methods)."""
        return self.google_api_key


# Global settings instance
settings = Settings()
