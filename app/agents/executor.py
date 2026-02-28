from typing import Dict, Any, Callable, Awaitable, Optional
from app.agents.schemas import AgentInput, AgentOutput, ExecutionPlan
from app.core.logging import logger

class PlanExecutor:
    """
    Iterates through a generated ExecutionPlan and calls the backend tools asynchronously.
    """
    
    def __init__(self, backend_execution_func: Callable[[str, Dict[str, Any], str, Optional[str]], Awaitable[Dict[str, Any]]]):
        """
        Inject the callable backend function to decouple the agent layer from httpx or FastAPI services.
        Signature: (route, parameters, auth_header, user_id) -> dict
        """
        self.backend_execution_func = backend_execution_func
    
    async def execute(self, plan: ExecutionPlan, input_context: AgentInput) -> AgentOutput:
        """Executes a structured plan step-by-step securely against the backend."""
        logger.info(f"Starting execution of plan: {plan.goal_summary}")
        
        if not plan.is_executable or not plan.steps:
            return AgentOutput(
                status="failed",
                response="The generated plan is either empty or marked as non-executable.",
                data={"plan": plan.model_dump()}
            )

        execution_results: Dict[int, Any] = {}
        successful_steps = 0
        
        # NOTE: A more complex variant would process DAG dependencies here (using step.depends_on)
        # For simplicity, we execute sequentially by step order.
        ordered_steps = sorted(plan.steps, key=lambda x: x.step_id)
        
        for step in ordered_steps:
            logger.info(f"Executing Step {step.step_id}: Tool '{step.tool_name}'")
            
            # Use the injected backend execution function
            result = await self.backend_execution_func(
                step.tool_name, 
                step.parameters,
                input_context.auth_header,
                input_context.user_id
            )
            
            execution_results[step.step_id] = result
            
            if result and "error" in result:
                logger.warning(f"Step {step.step_id} failed: {result['error']}")
                return AgentOutput(
                    status="partial_failure",
                    response=f"Execution halted at step {step.step_id} due to tool error.",
                    data={"execution_trace": execution_results}
                )
            
            successful_steps += 1
            
        summary = f"Successfully executed all {successful_steps} steps of the plan."
        return AgentOutput(
            status="success",
            response=summary,
            data={"results": execution_results}
        )

