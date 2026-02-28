from abc import ABC, abstractmethod
from app.agents.schemas import AgentInput, AgentOutput

class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized AI Agents.
    Forces adherence to the async run() pattern.
    """
    
    @abstractmethod
    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        Executes the agent's core capability.
        
        Args:
            agent_input: The standardized AgentInput containing user details, message, and context.
            
        Returns:
            An AgentOutput containing the status, text response, and structured data.
        """
        pass
