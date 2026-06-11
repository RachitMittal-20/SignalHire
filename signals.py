import re
from datetime import datetime
from typing import Dict

from config import (
    CONSULTING_FIRMS,
    CURRENT_YEAR,
    JD_MUST_HAVES,
    JD_MUST_KEYWORDS,
    JD_NICE_TO_HAVES,
    JD_NICE_KEYWORDS,
    ML_AI_TITLE_KEYWORDS,
    PRODUCTION_SIGNALS,
    PROFICIENCY_MAP,
    REFERENCE_DATE,
    RETRIEVAL_SIGNALS,
    TIER_1_INSTITUTIONS,
    TIER_2_INSTITUTIONS,
)
from disqualify import parse_year, sort_career_chronologically


def _build_skill_dict(candidate: dict) -> Dict[str, float]:
    skills = candidate.get("skills", [])
    assessment_scores = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    )
    result = {}
    for skill in skills:
        name = skill.get("name", "").lower().strip()
        if not name:
            continue
        proficiency = skill.get("proficiency", "beginner").lower()
        weight = PROFICIENCY_MAP.get(proficiency, 0.1)

        if name in assessment_scores:
            score = float(assessment_scores[name]) / 100.0
            weight = max(weight, score)

        result[name] = weight
    return result


def _keyword_match_score(
    keywords: list, skill_dict: Dict[str, float], text_blob: str
) -> float:
    score = 0.0
    for kw in keywords:
        kw_lower = kw.lower().strip()
        for skill_name, skill_weight in skill_dict.items():
            if kw_lower in skill_name or skill_name in kw_lower:
                score = max(score, skill_weight)
        if kw_lower in text_blob.lower():
            score = max(score, 0.3)
    return score


def _scan_signals_in_text(text: str, signals: list) -> float:
    text_lower = text.lower()
    matches = sum(1 for s in signals if s.lower() in text_lower)
    if matches == 0:
        return 0.0
    return min(matches * 0.03, 0.10)


def compute_technical_fit(candidate: dict) -> float:
    skill_dict = _build_skill_dict(candidate)

    career_descriptions = " ".join(
        r.get("description", "") for r in candidate.get("career_history", [])
    )
    blob = (
        candidate.get("profile", {}).get("headline", "")
        + " "
        + candidate.get("profile", {}).get("summary", "")
        + " "
        + candidate.get("profile", {}).get("current_title", "")
        + " "
        + career_descriptions
    )

    must_score = 0.0
    for criterion, weight in JD_MUST_HAVES.items():
        keywords = JD_MUST_KEYWORDS.get(criterion, [criterion])
        criterion_score = _keyword_match_score(keywords, skill_dict, blob)
        must_score += weight * criterion_score

    prod_bonus = _scan_signals_in_text(career_descriptions, PRODUCTION_SIGNALS)
    retr_bonus = _scan_signals_in_text(career_descriptions, RETRIEVAL_SIGNALS)

    nice_score = 0.0
    for criterion, weight in JD_NICE_TO_HAVES.items():
        keywords = JD_NICE_KEYWORDS.get(criterion, [criterion])
        criterion_score = _keyword_match_score(keywords, skill_dict, blob)
        nice_score += weight * criterion_score

    raw = must_score + prod_bonus + retr_bonus + nice_score
    return min(raw, 1.0)


def _is_ml_ai_title(title: str) -> bool:
    t = title.lower().strip()
    return any(kw in t for kw in ML_AI_TITLE_KEYWORDS)


def _parse_company_size(size_str) -> int:
    if not size_str:
        return 0
    s = str(size_str).strip()
    if s == "1000+":
        return 1001
    if "-" in s:
        try:
            parts = s.split("-")
            return int(parts[1])
        except (ValueError, IndexError):
            return 0
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


# Checked most-senior first so "Senior Principal Engineer" resolves to
# principal (4), not senior (3). Word boundaries prevent "intern" from
# matching "international" or "internal".
_SENIORITY_LEVELS = [
    (6, ["vp", "vice president", "chief", "cto", "cio", "ceo"]),
    (5, ["head", "director"]),
    (4, ["principal", "staff"]),
    (3, ["senior", "lead", "manager", "architect", "sr"]),
    (1, ["junior", "associate", "fresher", "jr"]),
    (0, ["intern", "trainee"]),
]
_SENIORITY_PATTERNS = [
    (level, re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b"))
    for level, kws in _SENIORITY_LEVELS
]


def _title_seniority_level(title: str) -> int:
    t = title.lower().strip()
    for level, pattern in _SENIORITY_PATTERNS:
        if pattern.search(t):
            return level
    return 2


def _median_tenure_months(candidate: dict) -> float:
    durations = [
        r.get("duration_months", 0) or 0
        for r in candidate.get("career_history", [])
    ]
    if not durations:
        return 0.0
    durations.sort()
    n = len(durations)
    if n % 2 == 1:
        return float(durations[n // 2])
    return (durations[n // 2 - 1] + durations[n // 2]) / 2.0


def _has_upward_title_progression(candidate: dict) -> bool:
    career = sort_career_chronologically(candidate.get("career_history", []))
    if len(career) < 2:
        return False
    first_title = career[0].get("title", "")
    last_title = career[-1].get("title", "")
    if not first_title or not last_title:
        return False
    return _title_seniority_level(last_title) > _title_seniority_level(first_title)


def compute_career_quality(candidate: dict) -> float:
    score = 0.0

    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    non_consulting = False
    consulting_count = 0
    named_company_count = 0
    has_ml_title = False

    for role in career:
        company = (role.get("company", "") or "").lower().strip()
        if company:
            named_company_count += 1
            if any(firm in company for firm in CONSULTING_FIRMS):
                consulting_count += 1
            else:
                non_consulting = True

        if not has_ml_title and _is_ml_ai_title(role.get("title", "")):
            size = _parse_company_size(role.get("company_size", ""))
            if size >= 50:
                has_ml_title = True

    if not has_ml_title:
        current_title = candidate.get("profile", {}).get("current_title", "")
        if _is_ml_ai_title(current_title):
            has_ml_title = True

    if non_consulting:
        score += 0.30
    if has_ml_title:
        score += 0.15

    median_tenure = _median_tenure_months(candidate)
    if median_tenure >= 36:
        score += 0.25
    elif median_tenure >= 24:
        score += 0.20
    elif median_tenure >= 18:
        score += 0.05

    if _has_upward_title_progression(candidate):
        score += 0.20

    # Zero out only when every *named* company is a consulting firm —
    # profiles with missing company names must not be punished.
    if named_company_count > 0 and consulting_count == named_company_count:
        score = 0.0

    return min(score, 1.0)


def compute_availability_signal(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})

    score = 0.0

    open_to_work = signals.get("open_to_work_flag", False)
    if open_to_work:
        score += 0.25

    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(str(last_active_str)[:10], "%Y-%m-%d").date()
            ref_date = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()
            days_since = (ref_date - last_active).days
            if days_since <= 30:
                score += 0.20
            elif days_since <= 90:
                score += 0.12
            elif days_since <= 180:
                score += 0.06
        except (ValueError, TypeError):
            pass

    response_rate = signals.get("recruiter_response_rate", 0.0) or 0.0
    score += min(response_rate, 1.0) * 0.15

    avg_response_hours = signals.get("avg_response_time_hours", float("inf")) or float("inf")
    if avg_response_hours <= 4:
        score += 0.10
    elif avg_response_hours <= 24:
        score += 0.07
    elif avg_response_hours <= 72:
        score += 0.04
    else:
        score += 0.01

    interview_rate = signals.get("interview_completion_rate", 0.0) or 0.0
    score += min(interview_rate, 1.0) * 0.15

    apps = signals.get("applications_submitted_30d", 0) or 0
    if apps >= 3:
        score += 0.10
    elif apps >= 1:
        score += 0.05

    notice_period = signals.get("notice_period_days", 90) or 90
    if notice_period <= 30:
        score += 0.05
    elif notice_period <= 60:
        score += 0.035
    elif notice_period <= 90:
        score += 0.02
    else:
        score += 0.005

    return min(score, 1.0)


def _check_education_tier(candidate: dict) -> int:
    edu_entries = candidate.get("education", [])
    if not edu_entries:
        return 0
    raw_tiers = []
    for e in edu_entries:
        t = e.get("tier")
        if t is None:
            continue
        # Dataset uses strings like "tier_1".."tier_4" / "unknown".
        m = re.search(r"\d+", str(t))
        if m:
            raw_tiers.append(int(m.group(0)))
    if raw_tiers:
        best = min(raw_tiers)
        if best <= 2:
            return int(best)
    for entry in edu_entries:
        institution = (entry.get("institution", "") or "").lower().strip()
        for kw in TIER_1_INSTITUTIONS:
            if kw in institution:
                return 1
    for entry in edu_entries:
        institution = (entry.get("institution", "") or "").lower().strip()
        for kw in TIER_2_INSTITUTIONS:
            if kw in institution:
                return 2
    return 0


def compute_seniority_fit(candidate: dict) -> float:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    if 6 <= yoe <= 9:
        score = 1.0
    elif 4 <= yoe <= 5 or 10 <= yoe <= 12:
        score = 0.7
    elif yoe == 3 or 13 <= yoe <= 15:
        score = 0.4
    else:
        score = 0.1

    tier = _check_education_tier(candidate)
    if tier == 1:
        score = min(score + 0.05, 1.0)
    elif tier == 2:
        score = min(score + 0.02, 1.0)

    return score
