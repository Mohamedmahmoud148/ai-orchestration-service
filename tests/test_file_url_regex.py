"""
Tests for the bounded file-URL regex in agent.py (M2 fix).

The previous unbounded regex was a ReDoS risk. The bounded version must:
  - Still extract genuine file URLs
  - Reject obviously malformed input fast (no backtracking blow-up)
  - Cap its scan window (caller-side defense)
"""
import time

from app.agents.agent import _FILE_URL_PATTERN


def test_extracts_single_file_url():
    urls = _FILE_URL_PATTERN.findall(
        "Here is the syllabus: https://example.com/files/syllabus.pdf"
    )
    assert urls == ["https://example.com/files/syllabus.pdf"]


def test_extracts_multiple_file_urls():
    text = (
        "Lecture 1: https://example.com/a.pdf "
        "Lecture 2: https://example.com/b.docx "
        "Roster:    https://example.com/c.xlsx"
    )
    urls = _FILE_URL_PATTERN.findall(text)
    assert len(urls) == 3
    assert all(u.startswith("https://") for u in urls)


def test_recognises_all_supported_extensions():
    for ext in ("pdf", "xlsx", "xls", "docx", "csv", "png", "jpg"):
        urls = _FILE_URL_PATTERN.findall(f"see https://x.com/file.{ext} please")
        assert len(urls) == 1, f"Extension {ext} not recognised"


def test_case_insensitive_extension_match():
    urls = _FILE_URL_PATTERN.findall("Get https://x.com/file.PDF here")
    assert len(urls) == 1


def test_rejects_unsupported_extensions():
    urls = _FILE_URL_PATTERN.findall("Check https://x.com/page.html")
    assert urls == []


def test_redos_resistance_long_a_chain():
    """Pathological input that breaks naive regexes — must complete fast."""
    hostile = "https://" + "a" * 5_000 + ".pdf"
    t0 = time.perf_counter()
    _FILE_URL_PATTERN.findall(hostile)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"Possible ReDoS: took {elapsed*1000:.1f}ms"


def test_redos_resistance_alternation_attack():
    """Repeated near-matches without a successful close."""
    hostile = ("https://x.com/" + "ab" * 2000) + ".html"  # bad extension at end
    t0 = time.perf_counter()
    result = _FILE_URL_PATTERN.findall(hostile)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"Possible ReDoS: took {elapsed*1000:.1f}ms"
    assert result == []
