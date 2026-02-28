import asyncio
import json
from datetime import datetime
from app.agents.schemas import AgentInput
from app.agents.model_router import ModelRouter
from app.agents.planner import PlannerAgent
from app.agents.executor import PlanExecutor
from app.agents.modules.summarization import SummarizationModule
from app.agents.modules.regulation import RegulationModule
from app.agents.modules.exam_generation import ExamGenerationModule
from app.core.logging import logger

# --- MOCKS FOR TESTING ---
class MockModelRouter(ModelRouter):
    async def generate_structured_json(self, prompt: str, system_instruction: str, model_id: str = "gemini-2.5-flash") -> dict | None:
        if "execution steps" in system_instruction.lower():
            # Mocking the Planner's LLM response
            return {
                "goal_summary": "Summarize, check for compliance, and generate an exam.",
                "is_executable": True,
                "steps": [
                    {
                        "step_id": 1,
                        "action": "agent_module",
                        "module_name": "SummarizationModule",
                        "input_payload": {"message": "The user wants an overview of quantum computing and then an exam on it."},
                        "depends_on": [],
                        "condition": None
                    },
                    {
                        "step_id": 2,
                        "action": "agent_module",
                        "module_name": "RegulationModule",
                        "input_payload": {"message": "{{step_1.output}}"},
                        "depends_on": [1],
                        "condition": None
                    },
                    {
                        "step_id": 3,
                        "action": "agent_module",
                        "module_name": "ExamGenerationModule",
                        "input_payload": {"message": "Quantum Computing based on {{step_1.output}}"},
                        "depends_on": [2],
                        "condition": None
                    }
                ]
            }
        
        # Mock ExamGeneration
        if "exam generator" in system_instruction.lower():
            return {
                "questions": [
                    {"q": "What is a qubit?", "options": ["Classical bit", "Quantum state", "Both"], "answer": "Quantum state"}
                ]
            }
        return {"result": "mock structured json"}

    async def generate(self, prompt: str, system_instruction: str = "", model_id: str = "gemini-2.5-flash") -> str | None:
        if "summarization ai" in system_instruction.lower():
            return "Quantum computing uses qubits to perform operations logarithmically faster."
        if "compliance ai" in system_instruction.lower():
            return "SAFE - No issues detected."
        return "Mock plain text response"

async def mock_backend_execution(tool_name: str, parameters: dict, auth_header: str | None, user_id: str | None) -> dict:
    return {"result": f"Executed {tool_name} successfully."}

async def run_multi_step_example():
    print("--- Starting Multi-Step Example ---")
    
    # 1. Dependency Injection setup
    # In production, these would be real clients (genai.Client, AsyncOpenAI, etc)
    # We pass None here because we've overridden the methods in MockModelRouter
    # to avoid needing real API keys for this example test.
    router = MockModelRouter(gemini_client=None, openai_client=None, anthropic_client=None)
    
    # 2. Instantiate Modules (injected with router)
    summarization = SummarizationModule(router)
    regulation = RegulationModule(router)
    exam_gen = ExamGenerationModule(router)
    
    registry = {
        "SummarizationModule": summarization,
        "RegulationModule": regulation,
        "ExamGenerationModule": exam_gen
    }
    
    # 3. Instantiate Agents
    planner = PlannerAgent(model_router=router, ranker=None)
    executor = PlanExecutor(
        backend_execution_func=mock_backend_execution,
        model_router=router,
        module_registry=registry
    )
    
    input_context = AgentInput(
        message="Summarize the concept of quantum computing, check the summary for compliance, and then generate an exam on it.",
        user_id="test_user_123"
    )
    
    # 4. Generate Plan
    print("\n[PLANNING PHASE]")
    planner_output = await planner.run(input_context)
    plan = planner_output.data.get("plan")
    
    if plan:
        print(f"Goal: {plan.goal_summary}")
        for step in plan.steps:
            print(f"  [{step.step_id}] Action: {step.action} | Module: {step.module_name}")
            print(f"       Payload: {step.input_payload}")
            
        # 5. Execute Plan
        print("\n[EXECUTION PHASE]")
        executor_output = await executor.execute(plan, input_context)
        
        print(f"\nExecutor Status: {executor_output.status}")
        print("Detailed Results:")
        results = executor_output.data.get("results", {})
        for step_id, res in results.items():
            print(f"  Step {step_id}:")
            if "status" in res:
                print(f"    Status: {res['status']}")
            if "output" in res:
                print(f"    Output: {res['output']}")
            if "data" in res:
                print(f"    Data: {res['data']}")
    else:
        print("Failed to generate plan.")

if __name__ == "__main__":
    asyncio.run(run_multi_step_example())
