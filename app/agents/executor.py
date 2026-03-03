from typing import Dict, Any, Callable, Awaitable, Optional, TYPE_CHECKING, List
import re
from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan
from app.core.logging import logger
if TYPE_CHECKING:
    from app.agents.model_router import ModelRouter
    from app.agents.base_agent import BaseAgent

class PlanExecutor:
    """
    Iterates through a generated ExecutionPlan and calls the backend tools,
    models, or agent modules asynchronously.
    """
    
    def __init__(
        self, 
        backend_execution_func: Callable[[str, Dict[str, Any], str, Optional[str]], Awaitable[Dict[str, Any]]],
        model_router: Optional['ModelRouter'] = None,
        module_registry: Optional[Dict[str, 'BaseAgent']] = None
    ):
        """
        Inject execution capabilities to decouple from actual services.
        """
        self.backend_execution_func = backend_execution_func
        self.model_router = model_router
        self.module_registry = module_registry or {}
        
    def _interpolate_payload(self, payload: Dict[str, Any], results: Dict[int, Any]) -> Dict[str, Any]:
        """Replaces {{step_X.output}} in strings with actual results."""
        interpolated = {}
        pattern = re.compile(r"\{\{step_(\d+)\.output\}\}")
        
        for k, v in payload.items():
            if isinstance(v, str):
                match = pattern.search(v)
                if match:
                    step_id = int(match.group(1))
                    if step_id in results:
                        # Simple replacement if it's the exact string, otherwise string inject
                        if v.strip() == f"{{{{step_{step_id}.output}}}}":
                            interpolated[k] = results[step_id]
                        else:
                            interpolated[k] = v.replace(f"{{{{step_{step_id}.output}}}}", str(results[step_id]))
                    else:
                        interpolated[k] = v
                else:
                    interpolated[k] = v
            # Very basic recursive dictionary handling could go here
            else:
                interpolated[k] = v
        return interpolated
        
    def _evaluate_condition(self, condition: str, results: Dict[int, Any]) -> bool:
        """In a real safe environment, use ast.literal_eval or a safe expressions parser."""
        # For demonstration purposes, we'll do a basic check assuming condition is empty or simple
        if not condition:
            return True
        logger.warning(f"Condition evaluation skipped for safety: {condition}")
        return True # Default to true for now, would need safe eval
    
    async def execute(self, plan: ExecutionPlan, input_context: AgentInput) -> AgentOutput:
        """Executes a structured plan step-by-step securely."""
        logger.info(f"Starting execution of plan: {plan.goal_summary}")
        
        if not plan.is_executable:
            return AgentOutput(
                status="failed",
                response="The generated plan is marked as non-executable.",
                data={"plan": plan.model_dump()}
            )

        execution_results: Dict[int, Any] = {}
        pre_step_results: List[Any] = []

        # 1. Execute Pre-Execution Steps (Tool Gates)
        for pre_step in plan.pre_execution_steps:
            logger.info(f"Executing Pre-Execution Step: {pre_step.tool}")
            try:
                res = await self.backend_execution_func(
                    pre_step.tool,
                    pre_step.input_payload,
                    input_context.auth_header,
                    input_context.user_id
                )
                pre_step_results.append(res)
                
                if res and isinstance(res, dict) and "error" in res:
                    return AgentOutput(
                        status="failed",
                        response=f"Pre-execution step '{pre_step.tool}' failed: {res['error']}",
                        data={"pre_step_results": pre_step_results}
                    )
            except Exception as e:
                logger.error(f"Pre-execution step {pre_step.tool} failed: {e}")
                return AgentOutput(
                    status="failed",
                    response=f"Pre-execution step '{pre_step.tool}' crashed.",
                    data={"error": str(e)}
                )

        # 2. Specialized Intent Flow: generate_exam
        if plan.intent == "generate_exam" and plan.exam_params:
            return await self._execute_generate_exam(plan, input_context, pre_step_results)

        # 3. Standard Step Loop (Fallback / Generic Plans)
        if not plan.steps:
            return AgentOutput(
                status="success",
                response=plan.goal_summary,
                data={"pre_step_results": pre_step_results}
            )

        successful_steps = 0
        ordered_steps = sorted(plan.steps, key=lambda x: x.step_id)
        
        for step in ordered_steps:
            # Check conditions
            if step.condition and not self._evaluate_condition(step.condition, execution_results):
                logger.info(f"Skipping Step {step.step_id} due to condition -> False")
                execution_results[step.step_id] = {"skipped": True}
                continue
                
            logger.info(f"Executing Step {step.step_id} - Action: {step.action}")
            interpolated_payload = self._interpolate_payload(step.input_payload, execution_results)
            
            # ... (rest of implementation remains similar but wrapped in try/except)
            try:
                if step.action == "tool":
                    result = await self.backend_execution_func(
                        step.tool_name, interpolated_payload, input_context.auth_header, input_context.user_id
                    )
                elif step.action == "model":
                    prompt = interpolated_payload.get("prompt", str(interpolated_payload))
                    sys_inst = interpolated_payload.get("system_instruction", "")
                    text_result = await self.model_router.generate(prompt=prompt, system_instruction=sys_inst)
                    result = {"output": text_result}
                elif step.action == "agent_module":
                    module = self.module_registry[step.module_name]
                    mi = AgentInput(message=interpolated_payload.get("message"), user_id=input_context.user_id, auth_header=input_context.auth_header)
                    mo = await module.run(mi)
                    result = {"status": mo.status, "output": mo.response, "data": mo.data}
                
                execution_results[step.step_id] = result
                if result and isinstance(result, dict) and "error" in result:
                     return AgentOutput(status="partial_failure", response=f"Halted at step {step.step_id}.", data={"results": execution_results})
                successful_steps += 1
            except Exception as e:
                 return AgentOutput(status="failed", response=f"Step {step.step_id} failed.", data={"error": str(e)})

        return AgentOutput(status="success", response=f"Executed {successful_steps} steps.", data={"results": execution_results})

    async def _execute_generate_exam(self, plan: ExecutionPlan, input_context: AgentInput, pre_results: List[Any]) -> AgentOutput:
        """Specific workflow for complex exam generation."""
        params = plan.exam_params
        
        # A. Resolve subjectOfferingId
        subject_offering_id = params.subjectOfferingId
        if not subject_offering_id:
            # Try to find it in pre-results (from ResolveSubjectOffering)
            for res in pre_results:
                if isinstance(res, dict) and "subjectOfferingId" in res:
                    subject_offering_id = res["subjectOfferingId"]
                    break
        
        if not subject_offering_id:
            return AgentOutput(status="failed", response="Could not resolve subjectOfferingId.")

        # B. Generate Exam Content via LLM
        logger.info("PlanExecutor: generating exam content via Gemini")
        prompt = (
            f"Generate a {params.examType} exam for {params.subjectName} ({params.departmentName}, {params.collegeName}).\n"
            f"Batch: {params.batchName}. Number of questions: {params.numberOfQuestions}.\n"
            f"Focus on core curriculum topics."
        )
        sys_inst = (
            "You are a professional academic examiner. Generate a structured exam with clear questions.\n"
            "Return JSON matching this schema:\n"
            "{\n"
            "  \"title\": \"Exam Title\",\n"
            "  \"questions\": [\n"
            "    { \"questionText\": \"...\", \"correctAnswer\": \"...\", \"mark\": 5 }\n"
            "  ]\n"
            "}"
        )
        
        exam_json = await self.model_router.generate_structured_json(
            prompt=prompt,
            system_instruction=sys_inst,
            model_id="gemini-2.5-flash"
        )
        
        if not exam_json:
            return AgentOutput(status="failed", response="Gemini failed to generate valid exam JSON.")

        # C. Call Backend tool CreateGeneratedExam
        logger.info(f"PlanExecutor: calling CreateGeneratedExam for offering {subject_offering_id}")
        tool_payload = {
            "subjectOfferingId": subject_offering_id,
            "examData": exam_json,
            "handleStudentRandomization": (params.variationMode == "different_per_student")
        }
        
        backend_result = await self.backend_execution_func(
            "CreateGeneratedExam",
            tool_payload,
            input_context.auth_header,
            input_context.user_id
        )
        
        if backend_result and "error" in backend_result:
            return AgentOutput(status="failed", response=f"Backend failed to create exam: {backend_result['error']}")
            
        return AgentOutput(
            status="success",
            response=f"Successfully generated and created the {params.examType} exam for {params.subjectName}.",
            data={"backend_payload": tool_payload, "backend_result": backend_result}
        )

