from typing import Optional
from app.agents.base_agent import BaseAgent
from app.agents.schemas import AgentInput, AgentOutput
from app.agents.model_router import ModelRouter
from app.core.logging import logger

class SummarizationModule(BaseAgent):
    """
    A specific agent module for summarizing text.
    """
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
        self.system_instruction = "You are a Summarization AI. Provide a concise bulleted summary of the provided text."
        
    async def run(self, agent_input: AgentInput) -> AgentOutput:
        logger.info("SummarizationModule executing...")
        
        prompt = f"Summarize the following:\n\n{agent_input.message}"
        
        # We can default to a fast model like flash
        result_text = await self.model_router.generate(
            prompt=prompt,
            system_instruction=self.system_instruction,
            model_id="gemini-2.5-flash"
        )
        
        if not result_text:
            return AgentOutput(
                status="failed",
                response="SummarizationModule failed to generate a response."
            )
            
        return AgentOutput(
            status="success",
            response=result_text,
            data={"module": "SummarizationModule"}
        )
