"""Configuration module for the Ad Generator application."""

from .settings import settings
from .templates import (
    BACKGROUND_TEMPLATES,
    BackgroundTemplate,
    get_template,
    get_template_for_product,
    list_templates,
)

__all__ = [
    "settings",
    "BACKGROUND_TEMPLATES",
    "BackgroundTemplate",
    "get_template",
    "get_template_for_product",
    "list_templates",
]
