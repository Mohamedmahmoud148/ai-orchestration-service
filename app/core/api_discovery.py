"""
api_discovery.py

Downloads and caches the OpenAPI (Swagger) schema from the backend.
Applies Role-Based Endpoint Allowlist to prevent the AI from calling
destructive or unauthorized routes.

v2.0 — Full Swagger coverage with RBAC-aware allowlist.
"""
import ssl
import json
import httpx
from typing import Dict, List, Optional, Tuple, Set

from app.core.config import settings
from app.core.logging import logger

_cached_schema: Optional[str] = None
_allowed_endpoints: Set[Tuple[str, str]] = set()

# ── BLOCKED methods completely ────────────────────────────────────────────────
_BLOCKED_METHODS = {"delete", "put", "patch"}

# ── BLOCKED path prefixes (never exposed to AI regardless of method) ──────────
_BLOCKED_PREFIXES = (
    "/api/auth",          # Authentication — AI never handles login/logout
    "/api/dev",           # Developer/debug routes
    "/api/ai",            # Prevent self-loop (orchestrator calling itself)
    "/api/auditlogs",     # Internal audit — not for AI queries
    "/api/notification",  # Push notifications — not an AI tool
)

# ── SAFE POST routes — only specific POST paths are allowed ───────────────────
# Read: AI may call these POST endpoints (creation/bulk/tool actions)
_SAFE_POST_PATHS = (
    # Exams
    "/api/exams",
    "/api/exams/generate-ai",
    "/api/exams/upload-pdf",
    "/api/exams/grade-submission",
    "/api/exams/",          # covers /api/exams/{id}/submit, /api/exams/{id}/auto-grade

    # Complaints (via ai-tools)
    "/api/ai-tools/create-complaint",
    "/api/ai-tools/distribute-exams",
    "/api/ai-tools/bulk-create-students",
    "/api/ai-tools/bulk-upload-grades",

    # Attendance
    "/api/attendance/sessions",
    "/api/attendance/check-in",

    # Enrollment
    "/api/enrollments/",
    "/api/enrollment/upload",

    # GPA recalculate
    "/api/gpa/student/",

    # Grades
    "/api/grades/calculate/",
    "/api/grades/",

    # Files
    "/api/file/upload",
    "/api/studentfiles/upload",
    "/api/materials/upload",

    # Students bulk
    "/api/students/bulk-upload-direct",
    "/api/students/bulk-upload-ai",
    "/api/students/import-excel",
)


def _is_allowed(path: str, method: str) -> bool:
    """Check if a path+method combination is allowed for AI usage."""
    method = method.lower()
    path_lower = path.lower()

    # Block destructive methods entirely
    if method in _BLOCKED_METHODS:
        return False

    # Block forbidden prefixes
    for prefix in _BLOCKED_PREFIXES:
        if path_lower.startswith(prefix):
            return False

    # GET requests: allow everything not blocked above
    if method == "get":
        return True

    # POST requests: must match a safe prefix
    if method == "post":
        for safe in _SAFE_POST_PATHS:
            if path_lower.startswith(safe.lower()):
                return True
        return False

    return False


async def fetch_and_filter_schema() -> None:
    """
    Downloads Swagger from the backend, filters out forbidden routes,
    and caches a compressed Schema String + precise Validation Set.
    """
    global _cached_schema, _allowed_endpoints

    base_url = (settings.BACKEND_BASE_URL or "").rstrip("/")
    if not base_url:
        logger.warning("api_discovery: BACKEND_BASE_URL is not configured.")
        return

    swagger_url = f"{base_url}/swagger/v1/swagger.json"
    logger.info("api_discovery: Fetching Swagger JSON from %s", swagger_url)

    try:
        transport = httpx.AsyncHTTPTransport(verify=False)
        async with httpx.AsyncClient(transport=transport, timeout=15.0) as client:
            response = await client.get(swagger_url)
            response.raise_for_status()
            data = response.json()

            paths = data.get("paths", {})
            schema_lines = []
            allowed_set = set()

            for path, methods in paths.items():
                for method, details in methods.items():
                    if not _is_allowed(path, method):
                        continue

                    summary = (
                        details.get("summary", "")
                        or details.get("description", "No description")
                    )
                    parameters = details.get("parameters", [])
                    param_names = [
                        p["name"]
                        for p in parameters
                        if p.get("in") in ("query", "path")
                    ]

                    # Store exact path for validation layer
                    allowed_set.add((method.upper(), path))

                    params_str = (
                        f" Params: {', '.join(param_names)}" if param_names else ""
                    )
                    schema_lines.append(
                        f"- {method.upper()} {path} → {summary}.{params_str}"
                    )

            if not schema_lines:
                logger.warning(
                    "api_discovery: Swagger fetched but NO routes passed filtering."
                )
                _cached_schema = "No allowed backend APIs available."
            else:
                _cached_schema = "\n".join(schema_lines)
                logger.info(
                    "api_discovery: Cached %d allowed endpoints.", len(allowed_set)
                )

            _allowed_endpoints = allowed_set

    except Exception as exc:
        logger.error("api_discovery: Failed to fetch Swagger schema: %s", exc)
        _cached_schema = "Backend API schema currently unavailable."


def get_allowed_endpoints_schema() -> str:
    """Returns the pre-filtered markdown string of available endpoints."""
    if _cached_schema is None:
        return "Backend API schema not loaded. Tell user to try again later."
    return _cached_schema


def validate_endpoint(method: str, endpoint: str) -> bool:
    """
    Validation Layer (CRITICAL):
    Determines if the LLM's requested endpoint is explicitly allowed.

    Handles path-parameter substitution:
      /api/Students/01KMXFB... matches /api/Students/{code}
    """
    if not _allowed_endpoints:
        logger.warning(
            "api_discovery.validate_endpoint: _allowed_endpoints is empty "
            "(Swagger not loaded at startup). Permitting %s %s under JWT-RBAC only.",
            method,
            endpoint,
        )
        return True

    target_parts = endpoint.strip("/").split("/")

    for allowed_method, allowed_path in _allowed_endpoints:
        if allowed_method != method.upper():
            continue

        allowed_parts = allowed_path.strip("/").split("/")
        if len(target_parts) != len(allowed_parts):
            continue

        match = True
        for tp, ap in zip(target_parts, allowed_parts):
            # Path param placeholder matches any value
            if ap.startswith("{") and ap.endswith("}"):
                continue
            if tp.lower() != ap.lower():
                match = False
                break

        if match:
            logger.debug(
                "api_discovery.validate_endpoint: ALLOWED %s %s", method, endpoint
            )
            return True

    logger.warning(
        "api_discovery.validate_endpoint: BLOCKED %s %s - not in allowlist",
        method,
        endpoint,
    )
    return False
