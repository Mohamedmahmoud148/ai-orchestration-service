"""
app/core/entity_tracker.py

Lightweight conversation entity extractor and tracker.

Purpose:
    Multi-turn conversations break when the user writes short follow-ups like:
    "شرحها", "وايه عن X؟", "هل عندي فيها درجة؟" — referring to something
    mentioned 2 turns ago.

    This module:
    1. Extracts academic entities from messages (course names, exam IDs, etc.)
    2. Persists them in memory so subsequent turns can resolve coreferences.
    3. Provides a plain-text entity block for injection into the LLM prompt.

Entity types tracked:
    course       — subject/course name (e.g. "Data Structures", "قواعد البيانات")
    exam         — exam title or ID
    doctor       — doctor/professor name
    department   — department name
    semester     — semester/term reference
    gpa          — GPA value mentioned
    goal         — inferred user goal ("graduation", "improve_gpa", etc.)

Design:
    - Pure Python, no LLM calls (< 1ms).
    - Falls back gracefully when memory is unavailable.
    - Stored in memory_store as user:{id}:entities (TTL 2h).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.core.logging import logger


# ── Entity extraction patterns ────────────────────────────────────────────────

# Arabic academic subject keywords (the word before a course name)
# Greedy capture of Arabic/Latin words, stopping at Arabic punctuation or end of line.
_COURSE_TRIGGER_AR = re.compile(
    r"(?:مادة|ماده|مادده|كورس|المادة|موضوع|محاضرة|مادتي|كورسي)\s+"
    r"([ا-يA-Za-z][ا-يA-Za-z0-9 ]{2,40})(?=[،,\.\?؟\n]|$)",
    re.UNICODE,
)

# English course triggers — greedy up to punctuation
_COURSE_TRIGGER_EN = re.compile(
    r"(?:course|subject|module|class|lecture)\s+"
    r"([A-Za-z][A-Za-z0-9 ]{2,40}?)(?=[,\.\?\n]|$|(?:\s+\w+\s+)|(?:\s+please))",
    re.IGNORECASE,
)

# GPA values
_GPA_PATTERN = re.compile(r"\b(?:gpa|معدل|معدلي)\s*(?:is|=|:|\s)\s*([0-4]\.\d{1,2})\b", re.IGNORECASE)

# Doctor name patterns — use full Arabic Unicode block [؀-ۿ] so
# variants like أ (U+0623) and إ (U+0625) are captured (both < ا U+0627).
_DOCTOR_PATTERN_AR = re.compile(
    r"(?:دكتور|دكتوره|الدكتور|أستاذ|أستاذة|د\.)\s+"
    r"([؀-ۿ]{2,15}(?:\s+[؀-ۿ]{2,15}){0,2})",
    re.UNICODE,
)
_DOCTOR_PATTERN_EN = re.compile(
    r"(?:dr\.?|prof\.?|doctor|professor)\s+([A-Za-z\s]{3,30}?)(?:\s|$|,|\.|;)",
    re.IGNORECASE,
)

# Semester/term references
_SEMESTER_PATTERN = re.compile(
    r"(?:ترم|فصل|semester|term)\s*(?:ال)?\s*(\d+|أول|تاني|ثاني|أولى|أولي|first|second|third|1|2|3)",
    re.IGNORECASE | re.UNICODE,
)

# ── Goal inference ────────────────────────────────────────────────────────────

_GOAL_KEYWORDS: Dict[str, List[str]] = {
    "graduation": [
        "تخرج", "التخرج", "أتخرج", "هتخرج", "graduation", "graduate", "finish", "complete",
        "خلاص الجامعة", "اتخرج", "متخرج",
    ],
    "improve_gpa": [
        "أحسن معدل", "أرفع معدل", "رفع الـ gpa", "improve gpa", "better grades",
        "أحسن درجاتي", "رفع الدرجات", "أحسن درجات",
    ],
    "exam_prep": [
        "مذاكرة", "ذاكر", "تجهيز امتحان", "exam prep", "prepare for", "study for",
        "مذاكره", "استعداد امتحان", "مراجعة",
    ],
    "understand_topic": [
        "أفهم", "فهم", "اشرح", "شرح", "explain", "understand", "what is", "إيه هو",
    ],
    "registration": [
        "تسجيل", "سجل", "register", "enroll", "اشترك", "ابدأ",
    ],
}


def extract_entities(message: str) -> Dict[str, Any]:
    """
    Extract academic entities from a single user message.

    Returns a dict of entity type → list of values found.
    Empty lists mean no entities of that type were found.
    """
    entities: Dict[str, Any] = {
        "courses": [],
        "exams": [],
        "doctors": [],
        "semesters": [],
        "gpa_values": [],
        "goals": [],
    }

    # Courses (Arabic)
    for m in _COURSE_TRIGGER_AR.finditer(message):
        val = m.group(1).strip()
        if val and len(val) > 2:
            entities["courses"].append(val)

    # Courses (English)
    for m in _COURSE_TRIGGER_EN.finditer(message):
        val = m.group(1).strip()
        if val and len(val) > 2:
            entities["courses"].append(val)

    # Doctors (Arabic + English)
    for m in _DOCTOR_PATTERN_AR.finditer(message):
        val = m.group(1).strip()
        if val and len(val) > 2:
            entities["doctors"].append(val)
    for m in _DOCTOR_PATTERN_EN.finditer(message):
        val = m.group(1).strip()
        if val and len(val) > 2:
            entities["doctors"].append(val)

    # Semesters
    for m in _SEMESTER_PATTERN.finditer(message):
        entities["semesters"].append(m.group(0).strip())

    # GPA values
    for m in _GPA_PATTERN.finditer(message):
        entities["gpa_values"].append(m.group(1))

    # Goals — keyword matching
    msg_lower = message.lower()
    for goal, keywords in _GOAL_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in msg_lower:
                if goal not in entities["goals"]:
                    entities["goals"].append(goal)
                break

    # Deduplicate all lists
    for key in entities:
        if isinstance(entities[key], list):
            entities[key] = list(dict.fromkeys(entities[key]))

    return entities


def merge_entities(
    existing: Dict[str, Any],
    new: Dict[str, Any],
    max_per_type: int = 5,
) -> Dict[str, Any]:
    """
    Merge new entities into existing, keeping the most recent ones.
    New values appear at the front (most recent = highest priority).
    """
    merged: Dict[str, Any] = {}
    for key in set(list(existing.keys()) + list(new.keys())):
        ex_list = existing.get(key, []) if isinstance(existing.get(key), list) else []
        nw_list = new.get(key, []) if isinstance(new.get(key), list) else []
        # New items first, then existing (preserving recency)
        combined = list(dict.fromkeys(nw_list + ex_list))
        merged[key] = combined[:max_per_type]
    return merged


def build_entity_context_block(entities: Dict[str, Any]) -> str:
    """
    Format tracked entities into a compact plain-text block for LLM injection.

    Example output:
        ## كيانات مذكورة في المحادثة
        - مواد دراسية: قواعد البيانات، Data Structures
        - أهداف: graduation, improve_gpa
    """
    if not entities:
        return ""

    lines: List[str] = []

    courses = entities.get("courses", [])
    if courses:
        lines.append(f"- مواد دراسية مذكورة: {', '.join(courses[:5])}")

    doctors = entities.get("doctors", [])
    if doctors:
        lines.append(f"- دكاترة مذكورون: {', '.join(doctors[:3])}")

    semesters = entities.get("semesters", [])
    if semesters:
        lines.append(f"- ترمات مذكورة: {', '.join(semesters[:3])}")

    goals = entities.get("goals", [])
    if goals:
        goal_labels = {
            "graduation": "التخرج",
            "improve_gpa": "تحسين الـ GPA",
            "exam_prep": "التحضير للامتحانات",
            "understand_topic": "فهم موضوع",
            "registration": "التسجيل",
        }
        goal_ar = [goal_labels.get(g, g) for g in goals[:3]]
        lines.append(f"- أهداف المستخدم: {', '.join(goal_ar)}")

    if not lines:
        return ""

    return "## كيانات تم ذكرها في المحادثة (للسياق)\n" + "\n".join(lines)


def infer_followup_suggestions(
    entities: Dict[str, Any],
    last_intent: Optional[str],
    role: str = "student",
) -> List[str]:
    """
    Generate 2-3 contextually-aware follow-up suggestions based on tracked
    entities and the last intent. These are injected into the response
    footer to guide the conversation naturally.

    Returns a list of Arabic suggestion strings (questions/actions).
    """
    suggestions: List[str] = []

    goals = entities.get("goals", [])
    courses = entities.get("courses", [])

    if role in ("student",):
        # Goal-based suggestions
        if "graduation" in goals:
            suggestions.append("🎓 عرض خطة التخرج — الساعات الباقية والمواد المطلوبة")
        if "improve_gpa" in goals:
            suggestions.append("📈 تحليل نقاط ضعفي في المواد وكيف أرفع المعدل")
        if "exam_prep" in goals:
            suggestions.append("📚 جدولة مذاكرة للمواد المسجلة هذا الترم")

        # Intent-based suggestions
        if last_intent in ("result_query", "backend_api_query"):
            suggestions.append("📊 مقارنة درجاتي بمتوسط الدفعة")
        if last_intent in ("regulation",):
            suggestions.append("🗺️ عرض roadmap الأكاديمي الخاص بي بناءً على اللائحة")
        if last_intent in ("academic_advice",):
            suggestions.append("📋 خطوات عملية لتحسين وضعي في الترم الجاي")

        # Course-based
        if courses:
            suggestions.append(f"📖 شرح محتوى مادة {courses[0]} من المواد المدرجة")

    elif role in ("admin", "superadmin"):
        if last_intent in ("backend_api_query",):
            suggestions.append("📊 تقرير Analytics Dashboard للطلاب والأقسام")
            suggestions.append("⚠️ عرض الطلاب في خطر أكاديمي (At Risk)")

    elif role == "doctor":
        if last_intent in ("result_query", "backend_api_query"):
            suggestions.append("📊 مقارنة نتائج الطلاب بين الدفعات")
        suggestions.append("📝 إنشاء امتحان جديد للمادة الحالية")

    # Trim to max 3 and deduplicate
    return list(dict.fromkeys(suggestions))[:3]
