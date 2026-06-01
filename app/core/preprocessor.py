"""
app/core/preprocessor.py  —  Layer 0: Message Pre-Processor

Zero-cost (no LLM, no I/O) normalization that runs before every
classification request.

Responsibilities:
  1. Language detection  — ar / en / mixed
  2. Script detection    — arabic / latin / mixed / arabizi
  3. Arabizi transliteration  — "sajelny" → "سجلني"
  4. Arabic normalization     — remove diacritics, unify alef/teh-marbuta
  5. Produce a `clean_text` used for embedding (better vectors)

Why this matters:
  - The embedding model sees normalized text, improving cosine similarity
    between "سجلني" and "sajelny" or "اعمللي quiz عن ML"
  - The LLM classifier receives pre-translated Arabizi so its compact
    prompt doesn't need to handle every possible Latin-letter substitution
  - Language metadata is stored in Redis preferences to avoid re-detection
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ScriptType = Literal["arabic", "latin", "mixed", "arabizi"]

# ── Arabizi patterns (Arabic written in Latin letters) ────────────────────────
# These heuristics catch the most common Egyptian Arabizi patterns without
# false-positiving on genuine English text.
_ARABIZI_WORD_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(3ayez|3ayz|2ayez|3yz)\b", re.IGNORECASE),   # عايز
    re.compile(r"\b(sajelny|sajlny|sgelny|saglny)\b", re.IGNORECASE),  # سجلني
    re.compile(r"\b(eb3at|ib3at|ib3et|b3at)\b", re.IGNORECASE),  # ابعت
    re.compile(r"\b(mawad|mwad|mwad)\b", re.IGNORECASE),         # مواد
    re.compile(r"\b(emte7an|imte7an|imti7an)\b", re.IGNORECASE), # امتحان
    re.compile(r"\b(fel|fil|fl)\s+[a-zA-Z]", re.IGNORECASE),     # في ال
    re.compile(r"\b(ana|enta|enti|howa|hia)\b", re.IGNORECASE),  # pronouns
    re.compile(r"\b(mesh|msh|ma3|mesh|msh)\b", re.IGNORECASE),   # مش / ما
    re.compile(r"\b(3an|3la|3ala)\b", re.IGNORECASE),            # عن / على
    re.compile(r"\b(kol|kull|kel)\b", re.IGNORECASE),            # كل
    re.compile(r"\b(wala|walla|wela)\b", re.IGNORECASE),         # ولا
    re.compile(r"\b(leh|leih|ley)\b", re.IGNORECASE),            # ليه
    re.compile(r"\b(eih|eiih|eyh)\b", re.IGNORECASE),            # ايه
    re.compile(r"\b(ya|yaa)\s+[a-zA-Z]", re.IGNORECASE),        # يا [name]
    re.compile(r"\b(2olo|olo|uly)\b", re.IGNORECASE),            # قولو
]

# Digit-as-letter substitutions (Arabizi hallmark)
_ARABIZI_DIGIT_RE = re.compile(r"[23456789]")

# ── Arabizi word-level substitution map ──────────────────────────────────────
_ARABIZI_WORD_MAP: dict[str, str] = {
    # Enrollment
    "sajelny": "سجلني",
    "sajlny": "سجلني",
    "sgelny": "سجلني",
    "saglny": "سجلني",
    # Want/need
    "3ayez": "عايز",
    "3ayz": "عايز",
    "2ayez": "عايز",
    "3yz": "عايز",
    # Send
    "eb3at": "ابعت",
    "ib3at": "ابعت",
    "ib3et": "ابعت",
    "b3at": "ابعت",
    # Material/subjects
    "mawad": "مواد",
    "mwad": "مواد",
    # Exam
    "emte7an": "امتحان",
    "imte7an": "امتحان",
    "imti7an": "امتحان",
    # Prepositions / particles
    "fel": "في ال",
    "fil": "في ال",
    "fl": "في ال",
    "3an": "عن",
    "3la": "على",
    "3ala": "على",
    "kol": "كل",
    "kull": "كل",
    "kel": "كل",
    "wala": "ولا",
    "walla": "ولا",
    "wela": "ولا",
    "leh": "ليه",
    "leih": "ليه",
    "ley": "ليه",
    "eih": "ايه",
    "eiih": "ايه",
    "eyh": "ايه",
    "mesh": "مش",
    "msh": "مش",
    # Pronouns
    "ana": "أنا",
    "enta": "أنت",
    "enti": "أنتِ",
    "howa": "هو",
    "hia": "هي",
    # Common words
    "complaint": "شكوى",
    "complain": "شكوى",
    "exam": "امتحان",
    "quiz": "اختبار",
    "subject": "مادة",
    "course": "مادة",
    "register": "سجل",
    "enroll": "اسجل",
}

# Digit → Arabic letter replacements (character-level, Arabizi convention)
_DIGIT_LETTER_MAP: dict[str, str] = {
    "2": "أ",
    "3": "ع",
    "5": "خ",
    "6": "ط",
    "7": "ح",
    "8": "ق",
    "9": "ص",
}


@dataclass(frozen=True)
class PreprocessedMessage:
    """
    Immutable result of pre-processing a single user message.

    Fields:
      original          — the raw message as received
      clean_text        — normalized text best suited for embedding
      detected_lang     — "ar" | "en" | "mixed"
      script_type       — "arabic" | "latin" | "mixed" | "arabizi"
      is_arabizi        — True when Arabizi patterns detected
      arabizi_note      — short note for LLM context ("user wrote in Arabizi")
    """
    original: str
    clean_text: str
    detected_lang: str           # "ar" | "en" | "mixed"
    script_type: ScriptType
    is_arabizi: bool
    arabizi_note: str            # injected into LLM prompt when non-empty


def preprocess_message(message: str) -> PreprocessedMessage:
    """
    Run the full Layer-0 pre-processing pipeline on a user message.

    Steps:
      1. Script analysis (count Arabic vs. Latin characters)
      2. Arabizi detection (word patterns + digit substitutions)
      3. Arabizi → Arabic transliteration (word-level)
      4. Arabic normalization (diacritics, alef variants, teh marbuta)
      5. Return PreprocessedMessage with all metadata

    Pure Python — no I/O, no LLM calls.
    """
    original = message
    msg = message.strip()
    if not msg:
        return PreprocessedMessage(
            original=original, clean_text="", detected_lang="en",
            script_type="latin", is_arabizi=False, arabizi_note="",
        )

    # ── 1. Character-level script analysis ───────────────────────────────────
    arabic_chars = sum(1 for c in msg if "؀" <= c <= "ۿ")
    total_alpha  = sum(1 for c in msg if c.isalpha())
    arabic_ratio = arabic_chars / max(total_alpha, 1)

    # ── 2. Arabizi detection ──────────────────────────────────────────────────
    # Fire when:  Arabic ratio < 0.30 (mostly Latin)
    #         AND at least one Arabizi word pattern matches
    msg_lower = msg.lower()
    is_arabizi = False

    if arabic_ratio < 0.50:
        # Check word patterns
        for pat in _ARABIZI_WORD_PATTERNS:
            if pat.search(msg):
                is_arabizi = True
                break
        # Check digit-as-letter (strong signal)
        if not is_arabizi and _ARABIZI_DIGIT_RE.search(msg):
            # Only count as Arabizi if digits are used as word-characters
            # (e.g. "3ayez" not "chapter 3")
            digit_words = re.findall(r"\b\w*[2356789]\w*\b", msg)
            if digit_words:
                is_arabizi = True

    # ── 3. Script type and language ───────────────────────────────────────────
    has_arabic = arabic_ratio > 0.20   # lower threshold captures mixed-lang messages
    has_latin  = (total_alpha - arabic_chars) / max(total_alpha, 1) > 0.20

    if is_arabizi:
        script_type: ScriptType = "arabizi"
        detected_lang = "ar"
    elif has_arabic and has_latin:
        script_type = "mixed"
        detected_lang = "mixed"
    elif has_arabic:
        script_type = "arabic"
        detected_lang = "ar"
    else:
        script_type = "latin"
        detected_lang = "en"

    # ── 4. Arabizi → Arabic transliteration ──────────────────────────────────
    working = msg
    arabizi_note = ""
    if is_arabizi or script_type == "mixed":
        working, arabizi_note = _transliterate_arabizi(working)

    # ── 5. Arabic normalization ───────────────────────────────────────────────
    clean = _normalize_arabic(working)

    return PreprocessedMessage(
        original=original,
        clean_text=clean,
        detected_lang=detected_lang,
        script_type=script_type,
        is_arabizi=is_arabizi,
        arabizi_note=arabizi_note,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _transliterate_arabizi(text: str) -> tuple[str, str]:
    """
    Apply Arabizi → Arabic substitutions.

    Returns (translated_text, note_for_llm).
    Note is injected into the classification prompt so the LLM knows
    the original language was Arabizi-encoded Arabic.
    """
    result = text

    # Word-level substitutions (case-insensitive, whole-word match)
    for arabizi_word, arabic_word in _ARABIZI_WORD_MAP.items():
        pattern = re.compile(rf"\b{re.escape(arabizi_word)}\b", re.IGNORECASE)
        result = pattern.sub(arabic_word, result)

    # Digit-level substitutions
    for digit, arabic_letter in _DIGIT_LETTER_MAP.items():
        # Only replace digits that appear within a word (not standalone numbers)
        result = re.sub(rf"(?<=\w){re.escape(digit)}|{re.escape(digit)}(?=\w)",
                        arabic_letter, result)

    changed = result.strip() != text.strip()
    note = "[User wrote in Arabizi — transliterated to Arabic for classification]" if changed else ""
    return result, note


def _normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for more consistent embedding.

      - Remove harakat (diacritics U+064B–U+065F)
      - Normalize alef variants (أإآ → ا)
      - Normalize teh marbuta (ة → ه)
      - Normalize waw with hamza (ؤ → و)
      - Normalize ya (ى → ي)
      - Collapse multiple spaces
    """
    # Diacritics
    text = re.sub(r"[ً-ٟ]", "", text)
    # Alef normalization
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Teh marbuta
    text = re.sub(r"ة", "ه", text)
    # Waw with hamza
    text = re.sub(r"ؤ", "و", text)
    # Ya without dots (alef maqsura)
    text = re.sub(r"ى", "ي", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language_fast(message: str) -> str:
    """
    Standalone language detector — returns "ar" | "en" | "mixed".
    Used by MemoryStore.detect_and_save_language (replaces the 20% heuristic).
    """
    result = preprocess_message(message)
    return result.detected_lang
