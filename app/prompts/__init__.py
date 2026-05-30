"""
app/prompts/

Externalized system prompts for the AI service.

Why externalize:
- Editing a 290-line string inside a .py file is dangerous (one stray `{` and
  the f-string blows up; one missed escape and the prompt is silently changed).
- Markdown files render in the IDE, support diff review, support PR comments
  on specific lines, and let non-engineers (instructors, domain experts)
  propose copy changes without touching Python code.
- Versioning per file makes it possible to A/B test or roll back a prompt
  without redeploying the whole service.

Loader contract:
    load_prompt("planner_system")        -> str   (raw text)
    render_prompt("dynamic_api_routing",  -> str   (substituted)
                   schema=..., message=...)

Placeholder syntax:
    Uses string.Template `$var` / `${var}` — same as Bash. Chosen over
    Jinja2 because:
      - zero new dependencies (stdlib only)
      - safe_substitute() never raises on missing vars (defense in depth)
      - no risk of {} in prompt prose being misread as format args

Versioning:
    Frontmatter at the top of each .md file is parsed into metadata:
        ---
        version: 1.0
        owner: ai-team
        last_reviewed: 2026-05-26
        ---

    Read via `prompt_metadata("planner_system")`.

Fallback behavior:
    If a prompt file is missing, load_prompt() raises FileNotFoundError
    with a clear message. Callers may catch this and fall back to a
    hardcoded constant during the migration window.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Dict, Tuple

from app.core.logging import logger

PROMPTS_DIR = Path(__file__).parent
_FRONTMATTER_DELIMITER = "---"


def _read_file(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Expected at {PROMPTS_DIR}/{name}.md"
        )
    return path.read_text(encoding="utf-8")


def _strip_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """
    Parse leading YAML-like frontmatter (single-level, string values only)
    and return (metadata_dict, body_text).

    Frontmatter shape:
        ---
        version: 1.0
        owner: ai-team
        ---
        actual prompt text starts here.

    If no frontmatter, returns ({}, original text).
    """
    if not text.startswith(_FRONTMATTER_DELIMITER):
        return {}, text

    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != _FRONTMATTER_DELIMITER:
        return {}, text

    meta: Dict[str, str] = {}
    body_start = None
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == _FRONTMATTER_DELIMITER:
            body_start = i + 1
            break
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()

    if body_start is None:
        return {}, text
    body = "\n".join(lines[body_start:]).lstrip("\n")
    return meta, body


@lru_cache(maxsize=64)
def _load_cached(name: str) -> Tuple[Dict[str, str], str]:
    """Read, parse frontmatter, cache the (metadata, body) tuple."""
    raw = _read_file(name)
    meta, body = _strip_frontmatter(raw)
    logger.info(
        "Prompt loaded: %s (version=%s, body_chars=%d)",
        name, meta.get("version", "?"), len(body),
    )
    return meta, body


def load_prompt(name: str) -> str:
    """Return the prompt body (without frontmatter). Cached."""
    _meta, body = _load_cached(name)
    return body


def prompt_metadata(name: str) -> Dict[str, str]:
    """Return the frontmatter metadata dict for the prompt."""
    meta, _body = _load_cached(name)
    return dict(meta)


def render_prompt(name: str, **variables: Any) -> str:
    """
    Load `name` and substitute `$var` / `${var}` placeholders.

    Uses string.Template.safe_substitute — missing vars are left as `$var`
    in the output rather than raising, so a prompt change that adds a
    placeholder doesn't crash production.
    """
    body = load_prompt(name)
    if not variables:
        return body
    return Template(body).safe_substitute(**variables)


def clear_prompt_cache() -> None:
    """Drop the LRU cache. Useful when admin endpoint reloads prompts."""
    _load_cached.cache_clear()
    logger.info("Prompt cache cleared")
