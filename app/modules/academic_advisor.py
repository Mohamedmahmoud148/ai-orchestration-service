"""
app/modules/academic_advisor.py  —  v2.0 Deep Orchestrator

Academic Advisor — intent: academic_advice

Pipeline (planner pattern):
  1. Pull the student's roadmap + overview from .NET backend (in parallel).
  2. Semantic-search the academic regulation (RAG) for passages relevant to
     the student's question + the conditions that define their situation
     (graduation requirements, retake rules, GPA thresholds, etc.).
  3. Compose a rich, comparative context: regulation rules + actual student
     state + relevant excerpts.
  4. Ask a strong LLM for a long, structured, personalised analysis that
     compares the student to the regulation and recommends specific actions.

This is the "ask the regulation and the student's record together" feature.
Per the user: this is the headline feature for the graduation defense — so
we deliberately spend tokens for depth and accuracy over speed.

Sources combined:
  GET  /api/Regulations/my-roadmap                 → semesters, passed/failed/
                                                      enrolled/upcoming, credit
                                                      hours, current GPA,
                                                      mustRetake, recommendedNext
  GET  /api/ai-tools/student-overview/{userId}     → grades, exams, subjects
  RAG  search_regulation(query)                    → relevant regulation passages

Architecture rules
------------------
- Backend is the single source of truth — AI never invents grade values.
- Regulation chunks are the single source of truth for academic rules.
- Long-form output (max_tokens=3000+) — this feature is allowed to be expensive.
- No direct DB access.
"""
from __future__ import annotations

import asyncio
import json as _json
from typing import Any, Dict, List, Optional

from app.agents.schemas import AgentInput, AgentOutput
from app.core.logging import logger
from app.prompts import load_prompt
from app.services.regulation_indexer import (
    index_all_active_regulations,
    is_any_regulation_indexed,
    search_regulation,
)


def _get_advisor_system_prompt() -> str:
    """Load from app/prompts/academic_advisor_v2.md; fall back to inline if missing."""
    try:
        return load_prompt("academic_advisor_v2")
    except Exception as exc:
        logger.warning(
            "AcademicAdvisor v2: prompt load failed — using inline fallback: %s", exc,
        )
        return _ADVISOR_SYSTEM_PROMPT_FALLBACK

_DEFAULT_MODEL = "openai/gpt-4o-mini"
_ADVISOR_MAX_TOKENS = 3000             # long, structured response
_RAG_TOP_K = 8
_RAW_CONTEXT_CHAR_BUDGET = 2500        # per backend section, before LLM

# GPA thresholds (kept for status labelling)
_GPA_WEAK_THRESHOLD      = 2.0
_GPA_AVERAGE_THRESHOLD   = 2.75
_GPA_GOOD_THRESHOLD      = 3.5


def _classify_gpa(gpa: Optional[float]) -> str:
    if gpa is None:
        return "unknown"
    if gpa < _GPA_WEAK_THRESHOLD:
        return "at_risk"
    if gpa < _GPA_AVERAGE_THRESHOLD:
        return "needs_improvement"
    if gpa < _GPA_GOOD_THRESHOLD:
        return "average"
    return "strong"


def _truncate_json(obj: Any, max_chars: int) -> str:
    """Compact JSON dump truncated to max_chars (preserves meaning over form)."""
    try:
        s = _json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " …[truncated]"


def _build_rag_queries(user_message: str, roadmap: Dict[str, Any]) -> List[str]:
    """
    Build a small set of RAG queries that together pull the regulation passages
    the LLM will need: the literal student question + structural rules
    (graduation, GPA, retake, prerequisites). Diversifying the queries gives
    the LLM more grounded context than relying on the user's phrasing alone.
    """
    queries: List[str] = [user_message]

    queries.extend([
        "متطلبات التخرج وعدد الساعات المعتمدة",
        "شروط النجاح والرسوب وإعادة المادة",
        "نظام التقدير و GPA و معدلات الفصل",
        "المواد الإجبارية والاختيارية",
    ])

    # If the roadmap reveals failures or unfinished semesters,
    # add targeted queries about those rules.
    must_retake = roadmap.get("mustRetake") or []
    if must_retake:
        queries.append("شروط إعادة المواد الراسبة")
    if (roadmap.get("currentGpa") or 0) and float(roadmap.get("currentGpa") or 0) < 2.0:
        queries.append("الإنذار الأكاديمي والفصل من الكلية")
    return queries


async def _gather_regulation_passages(
    user_message: str,
    roadmap: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run several RAG searches in parallel and merge unique chunks."""
    queries = _build_rag_queries(user_message, roadmap)
    results_per_query = await asyncio.gather(
        *(search_regulation(q, top_k=_RAG_TOP_K // 2) for q in queries),
        return_exceptions=True,
    )

    seen_ids: set[str] = set()
    merged: List[Dict[str, Any]] = []
    for res in results_per_query:
        if isinstance(res, Exception) or not res:
            continue
        for chunk in res:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

    # Keep top-N by score across all merged chunks
    merged.sort(key=lambda c: c.get("score", 0), reverse=True)
    return merged[:_RAG_TOP_K + 4]


_ADVISOR_SYSTEM_PROMPT_FALLBACK = """\
أنت "مرشد" — مرشد أكاديمي شخصي للطالب في الجامعة. مهمتك تحليل وضع الطالب الأكاديمي وتقديم استشارة عميقة ومحددة، مبنية حصرياً على:
1. بيانات الطالب الفعلية من السيستم (الـ roadmap + الـ overview).
2. مقاطع رسمية من اللائحة الأكاديمية.

🔴 قواعد لا استثناء فيها:
- ممنوع تخترع أي رقم، اسم مادة، أو شرط مش موجود في البيانات أو المقاطع المقدمة.
- لما تذكر شرط من اللائحة، اقتبس النص أو لخصه بأمانة مع الإشارة لمصدره.
- لو الطالب سأل عن حاجة مش متوفر بيانات عنها، قول كده بصراحة بدل التخمين.
- لغة الرد بنفس لغة سؤال الطالب: عربي عامية مع عامية، فصحى مع فصحى، إنجليزي مع إنجليزي.

🎯 شكل الرد المطلوب (مفصّل وطويل — الطالب عايز فهم عميق):

## 📊 تقييم وضعك الحالي
- الفصل الحالي، عدد الساعات المُنجزة من إجمالي ساعات اللائحة، نسبة التقدم.
- الـ GPA الحالي وتصنيفه (ممتاز / جيد / يحتاج تحسين / في خطر) — قارنه بمعايير اللائحة لو متوفرة.
- ملخص سريع للمواد: ناجح فيها، راسب، مسجل، ومتبقية.

## ⚠️ نقاط الخطر والمواد الحرجة
- المواد اللي لازم تعيدها (mustRetake) مع أسبابها من اللائحة.
- المواد المسجل فيها حالياً واللي محتاجة تركيز.
- لو الـ GPA قريب من حد الإنذار → نبهه بوضوح واذكر شرط الإنذار من اللائحة.

## 🛣️ خطة المسار الأكاديمي
- المواد المُقترحة الترم الجاي (من recommendedNext) ولماذا هي مناسبة.
- ترتيب أولوية: إيه اللي يخلصه الأول؟ المواد الإجبارية أم الاختيارية؟
- لو في prerequisites مذكورة في اللائحة → اشرحها.

## 📚 توصيات دراسية محددة
- لكل مادة حرجة، اقترح أسلوب دراسة عملي (مش كلام عام زي "ذاكر كويس").
- اربط بقواعد التقدير في اللائحة: كم درجة محتاج عشان يحسن وضعه؟

## 🎓 احتمالات التخرج
- بناءً على معدله الحالي والساعات الباقية، تقدير زمني للتخرج.
- لو في شروط للتخرج (CGPA معين، عدد ساعات، إلخ) في اللائحة، اذكرها وقارنها بحالته.

## 💬 الخطوات التالية
- 3 خطوات عملية محددة يقدر يبدأها من النهاردة.

ملاحظات الأسلوب:
- استخدم نقاط، جداول قصيرة لو ساعدت في التوضيح، **bold** للأرقام والشروط المهمة.
- emoji محسوب — مش في كل سطر.
- نبرة احترافية بس ودودة. ممنوع جمل روبوتية ("لا تتردد..." / "يسعدني...").
- لو سؤال الطالب محدد جداً (مثلاً: "هل أقدر أتخرج هذا الترم؟") → ركز إجابتك حول ذلك بدلاً من تغطية كل الأقسام.\
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Module
# ──────────────────────────────────────────────────────────────────────────────


class AcademicAdvisorModule:
    """
    Deep academic advisor — combines:
      • Live roadmap + overview from .NET backend
      • Semantic-searched regulation chunks
      • Long-form LLM analysis (max_tokens=3000)
    """

    def __init__(self, model_router, backend_client):
        self.model_router   = model_router
        self.backend_client = backend_client

    # ────────────────────────────────────────────────────────────────────────
    #  Data gathering (run in parallel)
    # ────────────────────────────────────────────────────────────────────────

    async def _fetch_roadmap(self, auth_header: Optional[str]) -> Optional[Dict[str, Any]]:
        """GET /api/Regulations/my-roadmap — student's full curriculum status."""
        try:
            data = await self.backend_client.fetch(
                route="/api/Regulations/my-roadmap",
                auth_header=auth_header,
            )
            if isinstance(data, dict) and not data.get("_error"):
                return data
            logger.warning("AcademicAdvisor: my-roadmap returned error/empty")
        except Exception as exc:
            logger.warning("AcademicAdvisor: my-roadmap fetch failed (non-fatal) — %s", exc)
        return None

    async def _fetch_overview(
        self,
        user_id: Optional[str],
        auth_header: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """GET /api/ai-tools/student-overview/{userId} — grades + exams."""
        if not user_id:
            return None
        try:
            data = await self.backend_client.fetch(
                route=f"/api/ai-tools/student-overview/{user_id}",
                auth_header=auth_header,
            )
            if isinstance(data, dict) and not data.get("_error"):
                return data
        except Exception as exc:
            logger.warning("AcademicAdvisor: student-overview fetch failed (non-fatal) — %s", exc)
        return None

    # ────────────────────────────────────────────────────────────────────────
    #  Prompt assembly
    # ────────────────────────────────────────────────────────────────────────

    def _build_data_block(
        self,
        roadmap: Optional[Dict[str, Any]],
        overview: Optional[Dict[str, Any]],
        academic_ctx: Dict[str, Any],
    ) -> str:
        """Build the structured data section the LLM will reason over."""
        lines: List[str] = ["=== بيانات الطالب الفعلية من السيستم ==="]

        # Profile snapshot (from academic_context as a quick header)
        name   = academic_ctx.get("studentName") or academic_ctx.get("name")
        dept   = academic_ctx.get("departmentName")
        batch  = academic_ctx.get("batchName")
        if any((name, dept, batch)):
            lines.append("[الملف الشخصي]")
            if name:   lines.append(f"  الاسم: {name}")
            if dept:   lines.append(f"  القسم: {dept}")
            if batch:  lines.append(f"  الدفعة: {batch}")
            lines.append("")

        # Roadmap = the core academic state
        if roadmap:
            gpa = roadmap.get("currentGpa")
            gpa_status = _classify_gpa(float(gpa)) if gpa is not None else "unknown"
            lines.append("[الـ Roadmap الأكاديمي]")
            lines.append(f"  اللائحة: {roadmap.get('regulationTitle', '—')}")
            lines.append(f"  الكلية: {roadmap.get('collegeName', '—')}")
            lines.append(f"  القسم: {roadmap.get('departmentName', '—')}")
            lines.append(f"  الدفعة: {roadmap.get('batchName', '—')}")
            lines.append(f"  إجمالي الفصول: {roadmap.get('totalSemesters', '—')}")
            lines.append(
                f"  إجمالي الساعات: {roadmap.get('totalCreditHours', '—')}, "
                f"منجز: {roadmap.get('completedCreditHours', '—')}, "
                f"باقي: {roadmap.get('remainingCreditHours', '—')}"
            )
            lines.append(
                f"  مواد: ناجح={roadmap.get('passedSubjects', 0)}, "
                f"راسب={roadmap.get('failedSubjects', 0)}, "
                f"مسجل حالياً={roadmap.get('currentlyEnrolled', 0)}"
            )
            lines.append(f"  GPA: {gpa if gpa is not None else 'غير متاح'} ({gpa_status})")

            # Per-semester breakdown (compact)
            sems = roadmap.get("semesters") or []
            if sems:
                lines.append("\n[تفصيل الفصول]")
                for s in sems[:12]:
                    if not isinstance(s, dict):
                        continue
                    lines.append(
                        f"  ترم {s.get('semesterNumber')}: حالة={s.get('status')}, "
                        f"ناجح={s.get('passedSubjects', 0)}/{s.get('totalSubjects', 0)}, "
                        f"راسب={s.get('failedSubjects', 0)}, "
                        f"مسجل={s.get('enrolledSubjects', 0)}, "
                        f"ساعات={s.get('earnedCreditHours', 0)}/{s.get('totalCreditHours', 0)}"
                    )
                    # Top-3 subjects of the semester for context
                    subs = s.get("subjects") or []
                    for sub in subs[:6]:
                        if not isinstance(sub, dict):
                            continue
                        grade_str = ""
                        if sub.get("gradeLetter"):
                            grade_str = f" — {sub['gradeLetter']} ({sub.get('finalScore', '?')})"
                        lines.append(
                            f"    • {sub.get('subjectName', '—')} ({sub.get('subjectCode', '')}) "
                            f"[{sub.get('status')}, {sub.get('creditHours', '?')} ساعة"
                            f"{', إجباري' if sub.get('isRequired') else ''}]"
                            f"{grade_str}"
                        )

            # Must-retake list (critical)
            must_retake = roadmap.get("mustRetake") or []
            if must_retake:
                lines.append("\n[⚠️ مواد لازم يعيدها — راسب فيها وهي إجبارية]")
                for sub in must_retake[:10]:
                    if isinstance(sub, dict):
                        lines.append(
                            f"  • {sub.get('subjectName', '—')} "
                            f"({sub.get('creditHours', '?')} ساعة) "
                            f"— درجته السابقة: {sub.get('gradeLetter', '?')}"
                        )

            # Recommended next semester
            rec_next = roadmap.get("recommendedNext") or []
            if rec_next:
                lines.append("\n[المواد المقترحة للترم الجاي]")
                for sub in rec_next[:10]:
                    if isinstance(sub, dict):
                        lines.append(
                            f"  • {sub.get('subjectName', '—')} "
                            f"({sub.get('creditHours', '?')} ساعة"
                            f"{', إجباري' if sub.get('isRequired') else ', اختياري'})"
                        )
        else:
            lines.append("[الـ Roadmap الأكاديمي] غير متاح حالياً.")

        # Overview = grades + exams history
        if overview:
            lines.append("\n[نتائج وامتحانات سابقة]")
            grades = overview.get("grades") or []
            if grades:
                lines.append(f"  درجات نهائية ({len(grades)}):")
                for g in grades[:15]:
                    if isinstance(g, dict):
                        lines.append(
                            f"    • {g.get('subjectName', '—')} "
                            f"({g.get('subjectCode', '')}): "
                            f"{g.get('gradeLetter', '?')} / {g.get('finalScore', '?')}"
                        )
            exams = overview.get("exams") or []
            if exams:
                lines.append(f"  امتحانات ({len(exams)}):")
                for e in exams[:10]:
                    if isinstance(e, dict):
                        graded = "✓" if e.get("isGraded") else "في الانتظار"
                        lines.append(
                            f"    • {e.get('examTitle', '—')} "
                            f"({e.get('subjectName', '—')}): "
                            f"{e.get('score', '?')}/{e.get('totalMarks', '?')} [{graded}]"
                        )

        return "\n".join(lines)

    def _build_regulation_block(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """Format regulation passages as a numbered block for the LLM."""
        if not chunks:
            return (
                "=== مقاطع اللائحة الأكاديمية ===\n"
                "(لم يتم استرجاع مقاطع من اللائحة. اعتمد على بيانات الطالب الفعلية فقط، "
                "وقل بصراحة لو سؤاله يتطلب نص من اللائحة غير متاح حالياً.)"
            )

        lines = ["=== مقاطع مأخوذة من اللائحة الأكاديمية الرسمية ==="]
        for i, ch in enumerate(chunks[:12], 1):
            title = (ch.get("metadata") or {}).get("materialTitle") or "Regulation"
            content = (ch.get("content") or "").strip()
            if len(content) > 800:
                content = content[:800] + " …"
            lines.append(f"\n[مقطع {i} — {title}]")
            lines.append(content)
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ────────────────────────────────────────────────────────────────────────

    async def run(self, agent_input: AgentInput, plan=None) -> AgentOutput:
        ctx          = agent_input.context or {}
        model_id     = ctx.get("selected_model") or _DEFAULT_MODEL
        academic_ctx: Dict[str, Any] = ctx.get("academic_context", {}) or {}
        auth         = agent_input.auth_header

        user_id = (
            agent_input.user_id
            or academic_ctx.get("userId")
            or academic_ctx.get("studentId")
            or academic_ctx.get("profileId")
        )

        logger.info(
            "AcademicAdvisor v2: user=%s msg=%r",
            user_id, agent_input.message[:100],
        )

        # ── 0. Make sure regulations are indexed (first-run convenience) ──────
        try:
            if not await is_any_regulation_indexed():
                logger.info("AcademicAdvisor v2: no regulation indexed yet — triggering index")
                await index_all_active_regulations(auth_header=auth)
        except Exception as exc:
            logger.warning("AcademicAdvisor v2: auto-index failed (non-fatal) — %s", exc)

        # ── 1. Fetch backend data in parallel ─────────────────────────────────
        roadmap_task = asyncio.create_task(self._fetch_roadmap(auth))
        overview_task = asyncio.create_task(self._fetch_overview(user_id, auth))
        roadmap, overview = await asyncio.gather(roadmap_task, overview_task)

        # ── 2. RAG search regulation (multiple queries in parallel) ───────────
        regulation_chunks = await _gather_regulation_passages(
            agent_input.message, roadmap or {},
        )

        logger.info(
            "AcademicAdvisor v2: roadmap=%s overview=%s reg_chunks=%d",
            bool(roadmap), bool(overview), len(regulation_chunks),
        )

        # ── 3. Build the rich, comparative prompt ─────────────────────────────
        data_block       = self._build_data_block(roadmap, overview, academic_ctx)
        regulation_block = self._build_regulation_block(regulation_chunks)

        # If we somehow have neither real data nor regulation chunks, be honest
        # about the limitation rather than hallucinating an answer.
        if not roadmap and not overview and not regulation_chunks:
            return AgentOutput(
                status="success",
                response=(
                    "مش قادر أوصل لبيانات الطالب من السيستم ولا للائحة الأكاديمية دلوقتي. "
                    "ممكن تكون مشكلة مؤقتة في الاتصال. حاول تاني بعد لحظات، "
                    "ولو فضلت المشكلة كلم الدعم التقني."
                ),
                data={"module": "AcademicAdvisorModule", "version": "v2", "no_data": True},
            )

        # Recent conversation to resolve pronouns / short follow-ups
        raw_history = (ctx.get("history") or [])[-4:]
        history_block = ""
        if raw_history:
            hlines = []
            for t in raw_history:
                tr = t.get("role", "user")
                tc = str(t.get("content", "")).strip()
                if tc:
                    speaker = "المستخدم" if tr == "user" else "المساعد"
                    hlines.append(f"{speaker}: {tc[:300]}")
            if hlines:
                history_block = (
                    "\n=== السياق السابق للمحادثة ===\n"
                    + "\n".join(hlines)
                    + "\n=== نهاية السياق ===\n"
                )

        user_prompt = (
            f"سؤال الطالب الحالي: {agent_input.message}\n"
            f"{history_block}\n"
            f"{data_block}\n\n"
            f"{regulation_block}\n\n"
            "بناءً على بيانات الطالب الفعلية والمقاطع المقتبسة من اللائحة أعلاه، "
            "قدّم استشارة أكاديمية مفصّلة وعميقة تتبع الشكل المطلوب في تعليمات النظام. "
            "اربط بشكل صريح بين أرقام الطالب وشروط اللائحة (ساعات منجزة مقابل ساعات التخرج، "
            "GPA الحالي مقابل حد الإنذار، إلخ). كل توصية لازم تكون مبنية على بيانات حقيقية، مش تخمين."
        )

        # ── 4. LLM — long, structured response ────────────────────────────────
        try:
            advice = await self.model_router.generate(
                prompt=user_prompt,
                system_instruction=_get_advisor_system_prompt(),
                model_id=model_id,
                max_tokens=_ADVISOR_MAX_TOKENS,
            )
        except Exception as exc:
            logger.error("AcademicAdvisor v2: LLM call failed — %s", exc)
            return AgentOutput(
                status="failed",
                response="مش قادر أولد الاستشارة دلوقتي. حاول تاني بعد لحظات.",
            )

        if not advice:
            return AgentOutput(
                status="failed",
                response="ما طلعش رد من الموديل. ممكن تكرر السؤال؟",
            )

        gpa_val = None
        if roadmap and roadmap.get("currentGpa") is not None:
            try:
                gpa_val = float(roadmap["currentGpa"])
            except (TypeError, ValueError):
                gpa_val = None

        return AgentOutput(
            status="success",
            response=advice,
            data={
                "module":                "AcademicAdvisorModule",
                "version":               "v2",
                "gpa":                   gpa_val,
                "gpa_status":            _classify_gpa(gpa_val),
                "roadmap_loaded":        bool(roadmap),
                "overview_loaded":       bool(overview),
                "regulation_chunks":     len(regulation_chunks),
                "model_used":            model_id,
                "max_tokens":            _ADVISOR_MAX_TOKENS,
                "must_retake_count":     len((roadmap or {}).get("mustRetake") or []),
                "recommended_next_count": len((roadmap or {}).get("recommendedNext") or []),
            },
        )
