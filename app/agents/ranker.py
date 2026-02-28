from typing import Any, Dict, List
from app.agents.base_agent import BaseAgent
from app.agents.schemas import AgentInput, AgentOutput

class ToolRanker(BaseAgent):
    """
    Optional plugin agent that receives User Intent and filters the 
    global list of tools down to just the highly relevant ones.
    
    This increases LLM accuracy limits token usage when passing 
    tools to the Planner.
    """
    def __init__(self, available_tools: List[str]):
        """Inject available tools to eliminate dependency on tool_registry"""
        self.available_tools = available_tools
        
    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Filters tools.
        For an MVP: We perform simple keyword matching against agent_input.message.
        """
        all_intents = self.available_tools
        
        # Super simple mock heuristic: If the word 'status' is in the message, 
        # prioritize 'GetUserStatus'. Else, return all.
        ranked_tools: List[str] = []
        message_lower = agent_input.message.lower()
        
        if "status" in message_lower and "GetUserStatus" in all_intents:
            ranked_tools.append("GetUserStatus")
        if "schedule" in message_lower or "meeting" in message_lower:
            if "ScheduleMeeting" in all_intents:
                ranked_tools.append("ScheduleMeeting")
                
        # Fallback if no specific keyword triggered
        if not ranked_tools:
            ranked_tools = all_intents
            
        return AgentOutput(
            status="success",
            response=f"Ranked {len(ranked_tools)} tools for execution.",
            data={"ranked_tools": ranked_tools}
        )
