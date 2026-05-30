"""
Tests for app/schemas/ — Intent/Role enums + AcademicContext contract.
"""
import pytest

from app.schemas import AcademicContext, Intent, Role


class TestIntentEnum:

    def test_str_equivalence(self):
        """Intent.X must equal its string form so legacy code keeps working."""
        assert Intent.GENERAL_CHAT == "general_chat"
        assert Intent.ACADEMIC_ADVICE == "academic_advice"
        assert Intent.GENERATE_EXAM == "generate_exam"

    def test_is_valid_recognises_known_intents(self):
        assert Intent.is_valid("general_chat")
        assert Intent.is_valid("regulation")
        assert Intent.is_valid("action_execute")

    def test_is_valid_rejects_unknown(self):
        assert not Intent.is_valid("delete_database")
        assert not Intent.is_valid("")

    def test_values_returns_set(self):
        values = Intent.values()
        assert isinstance(values, set)
        assert "general_chat" in values
        assert len(values) == 15  # locks the catalog size — any add must update this

    def test_no_drift_with_rbac_module(self):
        """Every intent the RBAC module knows must be in the enum (and vice versa)."""
        from app.core.rbac import _STUDENT_ALLOWED, _DOCTOR_ALLOWED
        unknown_to_enum = (_STUDENT_ALLOWED | _DOCTOR_ALLOWED) - Intent.values()
        assert not unknown_to_enum, f"RBAC has intents not in Intent enum: {unknown_to_enum}"


class TestRoleEnum:

    def test_str_equivalence(self):
        assert Role.STUDENT == "student"
        assert Role.DOCTOR == "doctor"
        assert Role.ADMIN == "admin"
        assert Role.SUPERADMIN == "superadmin"

    def test_safe_coerces_known_string(self):
        assert Role.safe("doctor") == Role.DOCTOR
        assert Role.safe("admin") == Role.ADMIN

    def test_safe_defaults_to_student_for_unknown(self):
        assert Role.safe("unknown") == Role.STUDENT
        assert Role.safe("Hacker") == Role.STUDENT

    def test_safe_handles_none_and_empty(self):
        assert Role.safe(None) == Role.STUDENT
        assert Role.safe("") == Role.STUDENT


class TestAcademicContext:

    def test_empty_dict_does_not_raise(self):
        ctx = AcademicContext.model_validate({})
        assert ctx.effective_user_id() is None
        assert ctx.display_name() is None
        assert ctx.effective_gpa() is None

    def test_extra_fields_allowed(self):
        """Backend may add fields; we must not break."""
        ctx = AcademicContext.model_validate({
            "userId": "01XYZ",
            "newFieldFromBackend2026": "value",
            "anotherUnknownThing": [1, 2, 3],
        })
        assert ctx.userId == "01XYZ"

    def test_gpa_coerced_from_string(self):
        """Backend sometimes sends GPA as string."""
        ctx = AcademicContext.model_validate({"gpa": "3.45"})
        assert ctx.gpa == 3.45
        assert ctx.effective_gpa() == 3.45

    def test_gpa_coerced_from_int(self):
        ctx = AcademicContext.model_validate({"gpa": 3})
        assert ctx.gpa == 3.0

    def test_gpa_invalid_becomes_none(self):
        ctx = AcademicContext.model_validate({"gpa": "not a number"})
        assert ctx.gpa is None

    def test_effective_user_id_priority(self):
        """userId > studentId > doctorId > profileId."""
        ctx = AcademicContext.model_validate({
            "userId": "USER",
            "studentId": "STU",
            "doctorId": "DOC",
            "profileId": "PROF",
        })
        assert ctx.effective_user_id() == "USER"

        ctx = AcademicContext.model_validate({"studentId": "STU", "profileId": "PROF"})
        assert ctx.effective_user_id() == "STU"

        ctx = AcademicContext.model_validate({"profileId": "PROF"})
        assert ctx.effective_user_id() == "PROF"

    def test_display_name_priority(self):
        ctx = AcademicContext.model_validate({"studentName": "Mohamed", "fullName": "Mohamed Mahmoud"})
        assert ctx.display_name() == "Mohamed"

        ctx = AcademicContext.model_validate({"fullName": "Mohamed Mahmoud"})
        assert ctx.display_name() == "Mohamed Mahmoud"

    def test_effective_gpa_picks_first_present(self):
        ctx = AcademicContext.model_validate({"GPA": 3.5, "cgpa": 3.7})
        # gpa is None, GPA wins (first in priority list after gpa)
        assert ctx.effective_gpa() == 3.5

    def test_id_coercion_for_objects(self):
        """ULIDs sometimes arrive as wrapped objects."""
        class FakeUlid:
            def __str__(self):
                return "01FROMOBJ"
        ctx = AcademicContext.model_validate({"userId": FakeUlid()})
        assert ctx.userId == "01FROMOBJ"
