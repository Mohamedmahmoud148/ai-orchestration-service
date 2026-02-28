from typing import Dict, Any, Callable, Awaitable, Optional, TYPE_CHECKING
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
        
        if not plan.is_executable or not plan.steps:
            return AgentOutput(
                status="failed",
                response="The generated plan is either empty or marked as non-executable.",
                data={"plan": plan.model_dump()}
            )

        execution_results: Dict[int, Any] = {}
        successful_steps = 0
        
        ordered_steps = sorted(plan.steps, key=lambda x: x.step_id)
        
        for step in ordered_steps:
            # 1. Check conditions
            if step.condition and not self._evaluate_condition(step.condition, execution_results):
                logger.info(f"Skipping Step {step.step_id} due to condition -> False")
                execution_results[step.step_id] = {"skipped": True}
                continue
                
            logger.info(f"Executing Step {step.step_id} - Action: {step.action}")
            
            # 2. Interpolate variables from previous steps
            interpolated_payload = self._interpolate_payload(step.input_payload, execution_results)
            
            result = None
            
            # 3. Route Execution based on Action type
            try:
                if step.action == "tool":
                    if not step.tool_name:
                        raise ValueError(f"Step {step.step_id} action 'tool' requires 'tool_name'")
                    result = await self.backend_execution_func(
                        step.tool_name, 
                        interpolated_payload,
                        input_context.auth_header,
                        input_context.user_id
                    )
                
                elif step.action == "model":
                    if not self.model_router:
                        raise ValueError("ModelRouter not injected but 'model' action requested.")
                    # Let's assume input_payload has 'prompt' and optionally 'system_instruction'
                    prompt = interpolated_payload.get("prompt", str(interpolated_payload))
                    sys_inst = interpolated_payload.get("system_instruction", "")
                    
                    text_result = await self.model_router.generate(
                        prompt=prompt, 
                        system_instruction=sys_inst,
                        model_id=step.model_name or "gemini-2.5-flash"
                    )
                    result = {"output": text_result}
                    
                elif step.action == "agent_module":
                    if not step.module_name or step.module_name not in self.module_registry:
                        raise ValueError(f"Module '{step.module_name}' not found in registry.")
                        
                    module = self.module_registry[step.module_name]
                    # Create a downstream input
                    module_input = AgentInput(
                        message=interpolated_payload.get("message", str(interpolated_payload)),
                        user_id=input_context.user_id,
                        auth_header=input_context.auth_header,
                        context=interpolated_payload.get("context", {})
                    )
                    module_output = await module.run(module_input)
                    result = {
                        "status": module_output.status, 
                        "output": module_output.response,
                        "data": module_output.data
                    }
                    
                else:
                    raise ValueError(f"Unknown action type: {step.action}")
                    
                execution_results[step.step_id] = result
                
                # Halt on explicit tool error objects
                if result and isinstance(result, dict) and "error" in result:
                     logger.warning(f"Step {step.step_id} output error: {result['error']}")
                     return AgentOutput(
                         status="partial_failure",
                         response=f"Execution halted at step {step.step_id} due to tool error.",
                         data={"execution_trace": execution_results}
                     )
                     
                successful_steps += 1
                
            except Exception as e:
                 logger.error(f"Step {step.step_id} failed with exception: {str(e)}")
                 return AgentOutput(
                     status="failed",
                     response=f"Execution aborted at step {step.step_id} due to exception.",
                     data={"error": str(e), "trace": execution_results}
                 )
            
        summary = f"Successfully executed {successful_steps} steps of the plan."
        
        # Return the final step's output as the main response text if available
        final_response_text = summary
        if ordered_steps:
            last_step_id = ordered_steps[-1].step_id
            last_result = execution_results.get(last_step_id)
            if isinstance(last_result, dict) and "output" in last_result:
                final_response_text = last_result["output"]
                
        return AgentOutput(
            status="success",
            response=final_response_text,
            data={"results": execution_results}
        )

