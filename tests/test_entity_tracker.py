"""
tests/test_entity_tracker.py

Unit tests for the entity extraction and tracking module.
Tests cover: entity extraction, merging, context block generation,
follow-up suggestions, and edge cases.
"""
import pytest

from app.core.entity_tracker import (
    extract_entities,
    merge_entities,
    build_entity_context_block,
    infer_followup_suggestions,
)


# ── extract_entities ──────────────────────────────────────────────────────────

class TestExtractEntities:
    def test_extracts_arabic_course(self):
        entities = extract_entities("عايز أفهم مادة قواعد البيانات.")
        # Should extract at least the start of "قواعد البيانات"
        assert any("قواعد" in c for c in entities["courses"])

    def test_extracts_english_course(self):
        entities = extract_entities("explain course Data Structures.")
        assert any("Data" in c for c in entities["courses"])

    def test_extracts_graduation_goal(self):
        entities = extract_entities("أنا عايز أتخرج السنة دي")
        assert "graduation" in entities["goals"]

    def test_extracts_improve_gpa_goal(self):
        entities = extract_entities("كيف أرفع معدلي؟")
        assert "improve_gpa" in entities["goals"]

    def test_extracts_exam_prep_goal(self):
        entities = extract_entities("عايز أذاكر للامتحان")
        assert "exam_prep" in entities["goals"]

    def test_extracts_arabic_doctor(self):
        entities = extract_entities("الدكتور أحمد بيشرح المادة كويس")
        assert len(entities["doctors"]) > 0

    def test_extracts_english_doctor(self):
        entities = extract_entities("Dr. Smith is my advisor")
        assert any("Smith" in d for d in entities["doctors"])

    def test_extracts_semester(self):
        entities = extract_entities("مواد الترم الأول")
        assert len(entities["semesters"]) > 0

    def test_no_entities_in_empty_message(self):
        entities = extract_entities("")
        for key in ("courses", "doctors", "goals", "semesters"):
            assert entities[key] == []

    def test_no_false_positives_on_greeting(self):
        entities = extract_entities("مرحبا كيف حالك؟")
        assert entities["courses"] == []
        assert entities["goals"] == []

    def test_deduplication(self):
        entities = extract_entities("مادة قواعد البيانات. ومادة قواعد البيانات.")
        # Regardless of exact extracted string, should not have duplicates
        assert len(entities["courses"]) == len(set(entities["courses"]))


# ── merge_entities ────────────────────────────────────────────────────────────

class TestMergeEntities:
    def test_merge_keeps_both_courses(self):
        existing = {"courses": ["Data Structures"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        new = {"courses": ["Algorithms"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        merged = merge_entities(existing, new)
        assert "Algorithms" in merged["courses"]
        assert "Data Structures" in merged["courses"]

    def test_merge_new_items_appear_first(self):
        existing = {"courses": ["Old Course"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        new = {"courses": ["New Course"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        merged = merge_entities(existing, new)
        assert merged["courses"][0] == "New Course"

    def test_merge_respects_max_per_type(self):
        existing = {"courses": [f"Course {i}" for i in range(10)], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        new = {"courses": ["New Course"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        merged = merge_entities(existing, new, max_per_type=5)
        assert len(merged["courses"]) == 5

    def test_merge_deduplicates(self):
        existing = {"courses": ["Data Structures"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        new = {"courses": ["Data Structures"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        merged = merge_entities(existing, new)
        assert merged["courses"].count("Data Structures") == 1

    def test_merge_empty_inputs(self):
        merged = merge_entities({}, {})
        assert merged == {}


# ── build_entity_context_block ────────────────────────────────────────────────

class TestBuildEntityContextBlock:
    def test_returns_empty_string_for_empty_entities(self):
        assert build_entity_context_block({}) == ""

    def test_includes_course_names(self):
        entities = {"courses": ["قواعد البيانات"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        block = build_entity_context_block(entities)
        assert "قواعد البيانات" in block

    def test_includes_arabic_goal_label(self):
        entities = {"courses": [], "goals": ["graduation"], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        block = build_entity_context_block(entities)
        assert "التخرج" in block

    def test_includes_doctor_name(self):
        entities = {"courses": [], "goals": [], "doctors": ["أحمد"], "semesters": [], "gpa_values": [], "exams": []}
        block = build_entity_context_block(entities)
        assert "أحمد" in block

    def test_starts_with_section_header(self):
        entities = {"courses": ["X"], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        block = build_entity_context_block(entities)
        assert block.startswith("##")


# ── infer_followup_suggestions ────────────────────────────────────────────────

class TestInferFollowupSuggestions:
    def test_graduation_goal_suggests_graduation_plan(self):
        entities = {"courses": [], "goals": ["graduation"], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        suggestions = infer_followup_suggestions(entities, "general_chat", "student")
        assert any("تخرج" in s or "graduation" in s.lower() for s in suggestions)

    def test_result_query_intent_suggests_comparison(self):
        entities = {"courses": [], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        suggestions = infer_followup_suggestions(entities, "result_query", "student")
        assert len(suggestions) > 0

    def test_admin_role_suggests_analytics(self):
        entities = {"courses": [], "goals": [], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        suggestions = infer_followup_suggestions(entities, "backend_api_query", "admin")
        assert any("Analytics" in s or "analytics" in s.lower() or "تقرير" in s for s in suggestions)

    def test_max_three_suggestions(self):
        entities = {"courses": ["X"], "goals": ["graduation", "improve_gpa", "exam_prep"], "doctors": [], "semesters": [], "gpa_values": [], "exams": []}
        suggestions = infer_followup_suggestions(entities, "academic_advice", "student")
        assert len(suggestions) <= 3

    def test_empty_entities_no_crash(self):
        suggestions = infer_followup_suggestions({}, None, "student")
        assert isinstance(suggestions, list)
