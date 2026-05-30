"""
Tests for app/core/rbac.py — the single source of truth for intent permissions.

Why these tests matter:
    A silent change to ROLE_PERMISSIONS can suddenly let students trigger
    exam generation, or block doctors from grading. This is the most
    security-sensitive surface in the AI service.
"""
import pytest

from app.core.rbac import (
    is_allowed,
    get_denial_message,
    log_blocked_attempt,
)
from app.schemas import Intent, Role


# ── Permission matrix tests ────────────────────────────────────────────────


class TestStudentPermissions:
    """Students get personal/study intents but never bulk ops or exam creation."""

    def test_student_can_chat(self):
        assert is_allowed(Intent.GENERAL_CHAT, Role.STUDENT)

    def test_student_can_query_own_results(self):
        assert is_allowed(Intent.RESULT_QUERY, Role.STUDENT)

    def test_student_can_submit_complaint(self):
        assert is_allowed(Intent.COMPLAINT_SUBMIT, Role.STUDENT)

    def test_student_can_get_academic_advice(self):
        assert is_allowed(Intent.ACADEMIC_ADVICE, Role.STUDENT)

    def test_student_can_query_backend(self):
        assert is_allowed(Intent.BACKEND_API_QUERY, Role.STUDENT)

    def test_student_cannot_generate_exam(self):
        assert not is_allowed(Intent.GENERATE_EXAM, Role.STUDENT)

    def test_student_cannot_view_complaint_summary(self):
        assert not is_allowed(Intent.COMPLAINT_SUMMARY, Role.STUDENT)

    def test_student_cannot_bulk_process_files(self):
        assert not is_allowed(Intent.FILE_PROCESSING, Role.STUDENT)


class TestDoctorPermissions:
    """Doctors get exam creation, student summaries, but not bulk admin ops."""

    def test_doctor_can_generate_exam(self):
        assert is_allowed(Intent.GENERATE_EXAM, Role.DOCTOR)

    def test_doctor_can_view_complaint_summary(self):
        assert is_allowed(Intent.COMPLAINT_SUMMARY, Role.DOCTOR)

    def test_doctor_cannot_submit_complaint(self):
        assert not is_allowed(Intent.COMPLAINT_SUBMIT, Role.DOCTOR)

    def test_doctor_cannot_bulk_process_files(self):
        assert not is_allowed(Intent.FILE_PROCESSING, Role.DOCTOR)


class TestAdminPermissions:
    """Admin/superadmin get everything — no per-intent denial."""

    def test_admin_can_do_anything(self):
        for intent in Intent:
            assert is_allowed(intent, Role.ADMIN), (
                f"admin should be allowed {intent!r}"
            )

    def test_superadmin_can_do_anything(self):
        for intent in Intent:
            assert is_allowed(intent, Role.SUPERADMIN), (
                f"superadmin should be allowed {intent!r}"
            )


class TestRoleSafetyDefaults:
    """Unknown or missing role must default to STUDENT (most restrictive)."""

    def test_unknown_role_defaults_to_student_perms(self):
        # Student cannot generate exam — so unknown role also cannot
        assert not is_allowed(Intent.GENERATE_EXAM, "hacker_role")

    def test_empty_role_defaults_to_student_perms(self):
        assert not is_allowed(Intent.GENERATE_EXAM, "")


# ── Denial message tests ──────────────────────────────────────────────────


class TestDenialMessages:
    """Denial messages must be bilingual and actionable (no internal jargon)."""

    def test_student_exam_denial_is_bilingual(self):
        msg = get_denial_message(Intent.GENERATE_EXAM, Role.STUDENT)
        assert "Exam" in msg or "exam" in msg
        assert "امتحان" in msg or "إنشاء" in msg

    def test_denial_never_exposes_internal_jargon(self):
        """No 'RBAC', 'intent', 'role permission' jargon in user-facing messages."""
        msg = get_denial_message(Intent.GENERATE_EXAM, Role.STUDENT)
        forbidden_jargon = ["RBAC", "ROLE_PERMISSIONS", "is_allowed", "frozenset"]
        for term in forbidden_jargon:
            assert term not in msg, f"Denial leaked internal term: {term}"

    def test_unknown_intent_denial_falls_back_gracefully(self):
        msg = get_denial_message("totally_unknown_intent", Role.STUDENT)
        assert msg  # never empty
        assert len(msg) > 10


# ── Audit logging test ───────────────────────────────────────────────────


def test_log_blocked_attempt_does_not_raise(caplog):
    """Audit logging must never throw — it must observe failures, not cause them."""
    import logging
    caplog.set_level(logging.WARNING)

    log_blocked_attempt(
        intent=Intent.GENERATE_EXAM,
        role=Role.STUDENT,
        user_id="01TESTUSER",
        extra={"module": "test"},
    )

    # Confirm a WARNING with structured fields was emitted
    assert any(
        "RBAC_BLOCKED" in rec.getMessage() and "generate_exam" in rec.getMessage()
        for rec in caplog.records
    )


def test_log_blocked_attempt_handles_missing_user_id(caplog):
    """user_id=None should still produce a valid log line."""
    import logging
    caplog.set_level(logging.WARNING)
    log_blocked_attempt(intent=Intent.FILE_PROCESSING, role=Role.STUDENT, user_id=None)
    assert any("unknown" in rec.getMessage() for rec in caplog.records)
