import json
from typing import Optional, Protocol
from pydantic import ValidationError
from app.agents.base_agent import BaseAgent
from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan
from app.core.logging import logger

class MemoryStore(Protocol):
    """Protocol defining how the Planner retrieves historical context."""
    async def get_context(self, user_id: str | None) -> str: ...

class PlannerAgent(BaseAgent):
    """
    Advanced agent that generates an ExecutionPlan consisting of 
    multiple steps.
    """
    
    def __init__(self, model_router, ranker=None, memory: Optional[MemoryStore] = None):
        """Inject dependencies to enforce Clean Architecture."""
        self.model_router = model_router
        self.ranker = ranker
        self.memory = memory
        
        # The schema definition we want the LLM to adhere to
        self.output_schema_instruction = (
            "You are an AI Planner mapping a user request to a sequence of execution steps.\n"
            "Generate an execution plan satisfying the request.\n"
            "Available tools: {tools}\n"
            "Note: You can pass outputs from previous steps using {{step_X.output}} syntax in the input_payload.\n"
            "Format your output strictly as JSON matching this structure:\n"
            '{\n'
            '  "goal_summary": "Description of the plan",\n'
            '  "is_executable": true,\n'
            '  "steps": [\n'
            '    {\n'
            '      "step_id": 1,\n'
            '      "action": "tool", // "tool", "model", or "agent_module"\n'
            '      "tool_name": "NameOfTool", // If action is "tool"\n'
            '      "model_name": null, // If action is "model", e.g. "gemini-2.5-flash"\n'
            '      "module_name": null, // If action is "agent_module"\n'
            '      "input_payload": {"key1": "value1"},\n'
            '      "depends_on": [],\n'
            '      "condition": null // Optional python expression eval condition\n'
            '    }\n'
            '  ]\n'
            '}\n'
        )

    async def run(self, agent_input: AgentInput) -> AgentOutput:
        """
        1. Ranks tools (optional preprocessing if injected).
        2. Injects Memory context (from optional memory store).
        3. Prompts the LLM for an ExecutionPlan via injected ModelRouter.
        4. Parses and validates the returned plan.
        """
        logger.info(f"PlannerAgent starting generation for: {agent_input.user_id}")
        
        # 1. Get filtered tools from Ranker (if injected)
        available_tools = []
        if self.ranker:
            ranker_output = await self.ranker.run(agent_input)
            if ranker_output.data:
                available_tools = ranker_output.data.get("ranked_tools", [])
        
        # 2. Build instructions
        system_instruction = self.output_schema_instruction.replace(
            "{tools}", json.dumps(available_tools)
        )
        
        # 3. Add memory context if injected
        memory_context = ""
        if self.memory:
            past_context = await self.memory.get_context(agent_input.user_id)
            if past_context:
                memory_context = f"\nPast Context: {past_context}\n"
                
        full_prompt = f"{memory_context}Request: {agent_input.message}"
        
        # 4. Request LLM via injected Router
        raw_json_response = await self.model_router.generate_structured_json(
            prompt=full_prompt,
            system_instruction=system_instruction
        )
        
        if not raw_json_response:
             return AgentOutput(
                status="failed",
                response="Planner failed to generate a valid plan from the LLM.",
            )
             
        # 4. Validate output matches Pydantic Schema
        try:
            plan = ExecutionPlan(**raw_json_response)
            return AgentOutput(
                status="success",
                response=plan.goal_summary,
                data={"plan": plan}
            )
        except ValidationError as e:
            logger.error(f"ExecutionPlan validation failed: {e}")
            return AgentOutput(
                status="failed",
                response="The LLM returned a malformed plan structure.",
                data={"errors": e.errors()}
            )
