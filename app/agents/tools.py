"""
app/agents/tools.py — Tool Definitions for ReactAgent

Extracted from react_agent.py (Section 5 — God File Refactoring).
This module owns all OpenAI function-calling tool schemas.

Import in react_agent.py:
    from app.agents.tools import TOOL_SCHEMAS, _FOLLOWUP_AR, _FOLLOWUP_EN
"""
from __future__ import annotations

# ── Follow-up detection phrases ───────────────────────────────────────────────

_FOLLOWUP_AR = (
    "لخصه", "لخصلي", "لخص", "اقراه", "اقرا", "اقرأه", "اقرأ", "اشرحه", "اشرح الملف",
    "اشرح المحتوى", "اشرح المحتوي", "ماذا يحتوي", "ما يحتوي", "اعمل quiz",
    "اعمل امتحان", "استخرج العناوين", "ملخص", "فيه ايه", "فيه إيه",
    "محتواه", "محتواها", "عاوز تلخيص", "عايز تلخيص",
)

_FOLLOWUP_EN = (
    "summarize it", "summarize this", "read it", "read this", "explain it",
    "explain this", "make a quiz", "generate exam", "generate a quiz",
    "list headings", "what's inside", "what is inside",
)

# ── Tool Schemas (OpenAI function-calling format) ─────────────────────────────

_TOOL_CALL_API: dict = {
    "type": "function",
    "function": {
        "name": "call_backend_api",
        "description": (
            "Call any backend API endpoint to retrieve or submit university data. "
            "Use this for: student info, grades, schedules, enrollments, materials list, "
            "doctors, departments, colleges, complaints, exams. "
            "Always use GET for data retrieval, POST only for explicit actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method. Use GET for data retrieval, POST for actions.",
                },
                "path": {
                    "type": "string",
                    "description": "API path, e.g. /api/Enrollments/my-enrollments",
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters for GET or body for POST.",
                },
            },
            "required": ["method", "path"],
        },
    },
}

_TOOL_REGULATION: dict = {
    "type": "function",
    "function": {
        "name": "read_regulation_pdf",
        "description": (
            "Read and search the academic regulation / student handbook. "
            "Use ONLY for questions about: graduation requirements, credit hours, "
            "academic rules, subject requirements, study plan, دليل الطالب, اللائحة الأكاديمية. "
            "Do NOT use for lecture materials or uploaded files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in the regulation document.",
                },
            },
            "required": ["query"],
        },
    },
}

_TOOL_GENERATE_EXAM: dict = {
    "type": "function",
    "function": {
        "name": "generate_exam",
        "description": (
            "Generate an AI exam for a specific subject. "
            "Use when: doctor asks to create/generate exam questions. "
            "NOT for viewing existing exams — use call_backend_api for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject_offering_id": {
                    "type": "string",
                    "description": "The SubjectOffering ID to generate exam for.",
                },
                "num_questions": {
                    "type": "integer",
                    "description": "Number of questions (default 10).",
                    "default": 10,
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard", "mixed"],
                    "default": "mixed",
                },
                "question_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types: mcq, true_false, short_answer, essay",
                },
            },
            "required": ["subject_offering_id"],
        },
    },
}

_TOOL_ACADEMIC_ANALYSIS: dict = {
    "type": "function",
    "function": {
        "name": "academic_analysis",
        "description": (
            "Deep academic advisor analysis for a student. "
            "Use when: student asks about GPA, academic risk, graduation plan, "
            "failed subjects, credit hours progress, recommended next subjects, "
            "or asks 'وضعي الأكاديمي إيه؟', 'هل هتخرج؟', 'أنا في خطر؟', "
            "'ايه المواد الباقية؟', 'كيف أحسن معدلي؟'. "
            "Fetches roadmap + grades + regulation passages and returns a deep analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "What the student specifically wants to know.",
                    "enum": [
                        "graduation_readiness", "gpa_improvement", "failed_subjects",
                        "next_semester_plan", "academic_risk", "general",
                    ],
                },
                "question": {
                    "type": "string",
                    "description": "The student's specific question.",
                },
            },
            "required": ["focus"],
        },
    },
}

_TOOL_READ_MATERIAL: dict = {
    "type": "function",
    "function": {
        "name": "read_material_pdf",
        "description": (
            "Download and read the CONTENT INSIDE a specific PDF/PPT file whose URL is already known. "
            "ONLY use when: (1) you already have the file_url from a previous step, "
            "AND (2) the user asks to read/summarize/explain the file content. "
            "NEVER use to list materials — use call_backend_api for that. "
            "NEVER use for regulations — use read_regulation_pdf for that. "
            "If file_url is unknown, first call the backend API to get it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "Full URL of the file to read.",
                },
                "task": {
                    "type": "string",
                    "enum": ["summarize", "explain", "list_headings", "read"],
                    "description": "What to do with the file.",
                    "default": "summarize",
                },
                "question": {
                    "type": "string",
                    "description": "Specific question or focus for the task.",
                },
            },
            "required": ["file_url", "task"],
        },
    },
}

# ── Exported list of all tool schemas ─────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    _TOOL_CALL_API,
    _TOOL_REGULATION,
    _TOOL_GENERATE_EXAM,
    _TOOL_ACADEMIC_ANALYSIS,
    _TOOL_READ_MATERIAL,
]
