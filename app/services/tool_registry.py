from typing import Dict, Any, Optional
from app.core.logging import logger

class ToolRegistry:
    """
    Centralized registry mapping AI intents to specific internal Agent Modules.
    Instead of mapping intent -> backend URL, we now map Intent -> Module identifier/class.
    The Backend URL mapping logic is decoupled and handled by BackendClient natively inside modules.
    """
    def __init__(self):
        # Maps IntentName -> Module Name
        # Keys MUST match the planner's VALID_INTENTS exactly.
        self._registry: Dict[str, str] = {
            "generate_exam":   "exam_generation",
            "summarization":   "summarization",
            "file_extraction": "file_extraction",
            "result_query":    "result_query",
        }

    def get_module_for_intent(self, intent_name: str) -> Optional[str]:
        """Returns the module identifier for the given intent, or None if not found."""
        return self._registry.get(intent_name)

    def register_module(self, intent_name: str, module_name: str):
        """Register a new module dynamically if needed."""
        self._registry[intent_name] = module_name

# Singleton instance
tool_registry = ToolRegistry()
