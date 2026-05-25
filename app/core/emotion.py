"""
emotion.py — Keyword-based emotion detection for chat responses.

Returns one of: happy, sad, thinking, excited, confused, neutral
Used by the chat endpoint to drive frontend avatar animations.
"""

import re

_HAPPY = re.compile(
    r"مبروك|congrat|تهانينا|ممتاز|رائع|excellent|great job|well done|"
    r"perfect|أحسنت|بالتوفيق|good luck|success|نجح|تفوق|achievement|إنجاز",
    re.IGNORECASE,
)

_SAD = re.compile(
    r"آسف|sorry|عذرا|للأسف|unfortunately|رسبت|failed|مشكلة|error|"
    r"خطأ|لم أتمكن|couldn't|cannot|لا يمكن|غير متاح|not available|"
    r"لا توجد|no results|لا أعرف|I don't know",
    re.IGNORECASE,
)

_EXCITED = re.compile(
    r"نتيجة|result|درجة|grade|score|امتحان|exam|اختبار|"
    r"ترتيب|rank|مرتبة|GPA|معدل|!{2,}",
    re.IGNORECASE,
)

_CONFUSED = re.compile(
    r"تقصد|تقصدي|تقصدون|did you mean|could you clarify|توضيح|"
    r"غير واضح|unclear|أي منهم|which one|تقصد أي",
    re.IGNORECASE,
)

_THINKING = re.compile(
    r"دعني|let me|سأبحث|looking|searching|processing|جاري|"
    r"سأتحقق|checking|بناءً على|based on|وفقاً|according to|"
    r"بالنظر إلى|considering",
    re.IGNORECASE,
)


def detect_emotion(text: str) -> str:
    """Return the dominant emotion inferred from response text."""
    if not text:
        return "neutral"

    if _CONFUSED.search(text):
        return "confused"
    if _SAD.search(text):
        return "sad"
    if _HAPPY.search(text):
        return "happy"
    if _EXCITED.search(text):
        return "excited"
    if _THINKING.search(text):
        return "thinking"
    return "neutral"
