from typing import Dict

class ToolRegistry:
    """
    Registry to map identified AI intents to specific backend endpoints.
    Allows easy expansion as new AI tools are added to the .NET backend.
    """
    def __init__(self):
        # Maps IntentName -> Route
        self._registry: Dict[str, str] = {
            "ScheduleMeeting": "/api/ai/execute/schedule-meeting",
            "GetUserStatus": "/api/ai/execute/user-status",
        }

    def get_route_for_intent(self, intent_name: str) -> str | None:
        """Returns the backend route for the given intent, or None if not found."""
        return self._registry.get(intent_name)

    def register_tool(self, intent_name: str, route: str):
        """Register a new tool dynamically if needed."""
        self._registry[intent_name] = route

# Singleton instance
tool_registry = ToolRegistry()
