from typing import Optional
from app.agents.base_agent import BaseAgent
from app.agents.schemas import AgentInput, AgentOutput
from app.agents.model_router import ModelRouter
from app.core.logging import logger

class RegulationModule(BaseAgent):
    """
    A specific agent module for checking text against compliance/regulation rules.
    """
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
        self.system_instruction = (
            "You are a Compliance AI. Check the provided text for any regulatory violations "
            "(e.g., PII leaks, inappropriate content, internal policy breaches). "
            "Reply with 'SAFE' or explicitly list the violations."
        )
        
    async def run(self, agent_input: AgentInput) -> AgentOutput:
        logger.info("RegulationModule executing...")
        
        # Typically regulation might need a stronger model
        result_text = await self.model_router.generate(
            prompt=agent_input.message,
            system_instruction=self.system_instruction,
            model_id="gpt-4o" 
        )
        
        if not result_text:
            return AgentOutput(
                status="failed",
                response="RegulationModule failed to generate a response."
            )
            
        is_safe = "SAFE" in result_text.upper()
            
        return AgentOutput(
            status="success" if is_safe else "flagged",
            response=result_text,
            data={"module": "RegulationModule", "is_safe": is_safe}
        )
