"""
api_discovery.py

Downloads and caches the OpenAPI (Swagger) schema from the backend.
Applies the Endpoint Allowlist Layer and Swagger Filtering Layer to prevent the AI
from seeing, suggesting, or calling administrative or destructive routes.
"""
import ssl
import json
import httpx
from typing import Dict, List, Optional, Tuple, Set

from app.core.config import settings
from app.core.logging import logger

_cached_schema: Optional[str] = None
_allowed_endpoints: Set[Tuple[str, str]] = set()

# ── 1. Endpoint Allowlist (MANDATORY Constraint) ─────────────────────────────
# We block all DELETE, PUT, PATCH methods completely.
_ALLOWED_METHODS = {"get", "post"}

# Block specific routes explicitly, even if they are GET or POST.
_BLOCKED_PREFIXES = (
    "/api/auth",             # Authentication endpoints
    "/api/dev",              # Developer test endpoints
    "/api/ai",               # Prevents orchestrator calling itself in a loop
    "/api/structure",        # Destructive DB reset routes if any exist here
)

# Safe POST prefixes. If it's a POST, it MUST match one of these explicitly.
_SAFE_POST_PREFIXES = (
    "/api/exams",            # Generating exams
    "/api/complaints",       # Submitting complaints
    "/api/files",            # Bulk uploads etc
)


def _is_allowed(path: str, method: str) -> bool:
    """Execution Validation Layer Constraint 1: Check against allowlist."""
    method = method.lower()
    path_lower = path.lower()

    if method not in _ALLOWED_METHODS:
        return False

    for prefix in _BLOCKED_PREFIXES:
        if path_lower.startswith(prefix):
            return False

    if method == "post":
        is_safe_post = any(path_lower.startswith(p) for p in _SAFE_POST_PREFIXES)
        if not is_safe_post:
            return False

    return True


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
        # Avoid SSL verification issues when fetching from self-signed internal endpoints
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
                    
                    # Optional user constraint: Check for [AI_ALLOWED] tag
                    # If tags exist, we could filter here. Currently not enforced 
                    # as all internal APIs without blocks should be mapped.
                    
                    summary = details.get("summary", "") or details.get("description", "No description")
                    parameters = details.get("parameters", [])
                    param_names = [p["name"] for p in parameters if p.get("in") == "query" or p.get("in") == "path"]
                    
                    # Store exact match for Validation Layer
                    allowed_set.add((method.upper(), path))
                    
                    params_str = f" Params: {', '.join(param_names)}" if param_names else ""
                    schema_lines.append(f"- {method.upper()} {path} → {summary}.{params_str}")

            if not schema_lines:
                logger.warning("api_discovery: Swagger fetched but NO routes passed filtering.")
                _cached_schema = "No allowed backend APIs available."
            else:
                _cached_schema = "\n".join(schema_lines)
                logger.info("api_discovery: Cached %d allowed endpoints.", len(allowed_set))

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
    """
    # The LLM might output `/api/Students/123`. 
    # Our allowed set contains `/api/Students/{id}`.
    # We must do basic path matching.
    target_parts = endpoint.strip("/").split("/")
    
    for allowed_method, allowed_path in _allowed_endpoints:
        if allowed_method != method.upper():
            continue
            
        allowed_parts = allowed_path.strip("/").split("/")
        if len(target_parts) != len(allowed_parts):
            continue
            
        match = True
        for tp, ap in zip(target_parts, allowed_parts):
            if ap.startswith("{") and ap.endswith("}"):
                continue # Param placeholder matches anything
            if tp.lower() != ap.lower():
                match = False
                break
                
        if match:
            return True
            
    return False

