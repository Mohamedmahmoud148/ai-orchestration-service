import asyncio
import os
from google import genai

async def test_planner():
    gemini_key = os.getenv("GEMINI_API_KEY", "AIzaSyDsnFfN-TD2WtPvkblvdHPIcdQpUJR4f0s")
    client = genai.Client(api_key=gemini_key)
    
    _SYSTEM_PROMPT = """You are an AI Planning Agent.

Your ONLY job is to read the user's message and return a single JSON object that
classifies the request.

Valid intents:
- general_chat       — casual conversation, questions, or anything else
- summarization      — the user wants a summary of a text or document
- generate_exam      — the user (a doctor/educator) wants an exam or quiz generated
- result_query       — the user wants to query academic results or grades
- file_extraction    — the user wants to extract or parse information from a file

Rules:
1. Reply with ONLY a raw JSON object. No markdown, no code fences, no extra text.
2. Use exactly this schema:

{
  "intent": "<one of the valid intents above>",
  "goal_summary": "<one short sentence describing what the user wants>",
  "is_executable": true,
  "exam_params": null,
  "pre_execution_steps": [],
  "steps": []
}

3. If the intent is generate_exam AND the user supplied exam details, populate
   exam_params with as many of these fields as you can extract:
     collegeName, departmentName, batchName, subjectName,
     numberOfQuestions (integer), examType ("midterm"|"final"),
     variationMode ("same_for_all"|"different_per_student"),
     subjectOfferingId (string | null)

4. If intent is anything other than generate_exam, keep exam_params as null.
5. If you lack the 'subjectOfferingId' for an exam, 'ResolveSubjectOffering' is a tool you can add to 'pre_execution_steps' to find it.
6. Never output anything outside the JSON object.
"""
    prompt = "User role: doctor\nUser message: عايز أعمل امتحان Midterm لمادة Math مكون من 10 أسئلة."
    print("Sending prompt to Gemini...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=_SYSTEM_PROMPT + "\n\n" + prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )
        print("Raw Response:", response.text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_planner())
