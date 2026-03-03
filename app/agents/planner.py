import json
from typing import Optional, Protocol
from pydantic import ValidationError
from app.agents.base_agent import BaseAgent
from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan, PreExecutionStep
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
            "\n"
            "## Intent: generate_exam\n"
            "When the request is about generating an exam, quiz, or assessment:\n"
            "  - Set \"intent\": \"generate_exam\" in the plan.\n"
            "  - Extract the following fields into \"exam_params\":\n"
            "      collegeName       (string, required)\n"
            "      departmentName    (string, required)\n"
            "      batchName         (string, required)\n"
            "      subjectName       (string, required)\n"
            "      numberOfQuestions (integer, required)\n"
            "      examType          (\"midterm\" | \"final\", required)\n"
            "      variationMode     (\"same_for_all\" | \"different_per_student\", required)\n"
            "      subjectOfferingId (string | null — include only when the user provides it)\n"
            "  - Do NOT generate exam content. Only build the plan.\n"
            "  - If subjectOfferingId is null or absent, you MUST add a pre_execution_steps entry:\n"
            "      {\n"
            "        \"tool\": \"ResolveSubjectOffering\",\n"
            "        \"reason\": \"subjectOfferingId is required to generate the exam but was not provided\",\n"
            "        \"input_payload\": {\n"
            "          \"collegeName\": <value>,\n"
            "          \"departmentName\": <value>,\n"
            "          \"batchName\": <value>,\n"
            "          \"subjectName\": <value>\n"
            "        }\n"
            "      }\n"
            "  - When subjectOfferingId is known, pre_execution_steps must be an empty array.\n"
            "\n"
            "Format your output strictly as JSON matching this structure:\n"
            '{\n'
            '  "goal_summary": "Description of the plan",\n'
            '  "intent": "generate_exam",  // or null for other intents\n'
            '  "is_executable": true,\n'
            '  "exam_params": { ... },  // populated when intent is generate_exam, else null\n'
            '  "pre_execution_steps": [],  // PreExecutionStep objects when required\n'
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

            # ── Deterministic guard for generate_exam ─────────────────────
            # If the LLM correctly identified the intent but forgot to inject
            # the ResolveSubjectOffering step, we inject it here so the
            # executor never receives an incomplete plan.
            if (
                plan.intent == "generate_exam"
                and plan.exam_params is not None
                and plan.exam_params.subjectOfferingId is None
            ):
                already_present = any(
                    s.tool == "ResolveSubjectOffering"
                    for s in plan.pre_execution_steps
                )
                if not already_present:
                    logger.info(
                        "PlannerAgent: injecting ResolveSubjectOffering pre-execution step "
                        "(subjectOfferingId not provided)"
                    )
                    plan.pre_execution_steps.append(
                        PreExecutionStep(
                            tool="ResolveSubjectOffering",
                            reason="subjectOfferingId is required to generate the exam but was not provided",
                            input_payload={
                                "collegeName": plan.exam_params.collegeName,
                                "departmentName": plan.exam_params.departmentName,
                                "batchName": plan.exam_params.batchName,
                                "subjectName": plan.exam_params.subjectName,
                            },
                        )
                    )
            # ──────────────────────────────────────────────────────────────

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
