import json
from typing import Optional
from app.agents.base_agent import BaseAgent
from app.agents.schemas import AgentInput, AgentOutput
from app.agents.model_router import ModelRouter
from app.core.logging import logger

class ExamGenerationModule(BaseAgent):
    """
    A specific agent module for generating exam questions from a topic.
    """
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
        self.system_instruction = (
            "You are an Exam Generator AI. Generate a list of 3 multiple-choice questions "
            "based on the provided topic. Return strict JSON with a root 'questions' array."
        )
        
    async def run(self, agent_input: AgentInput) -> AgentOutput:
        logger.info("ExamGenerationModule executing...")
        
        # Structured output requires generate_structured_json
        json_result = await self.model_router.generate_structured_json(
            prompt=f"Topic: {agent_input.message}",
            system_instruction=self.system_instruction,
            model_id="claude-3-opus" 
        )
        
        if not json_result:
            return AgentOutput(
                status="failed",
                response="ExamGenerationModule failed to generate JSON."
            )
            
        return AgentOutput(
            status="success",
            response="Generated exam questions successfully.",
            data={"module": "ExamGenerationModule", "questions": json_result.get("questions", [])}
        )
