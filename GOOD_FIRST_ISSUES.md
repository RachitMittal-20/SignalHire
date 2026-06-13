# 🎯 Good First Issues

Bite-sized, well-scoped tasks for new contributors. Each one touches a small
surface area and has a clear definition of done.

---

## 1. 🧹 Dead variable in `precompute.py`

**Difficulty:** trivial · **Files:** `precompute.py`

`id_batch` is appended to and reset alongside `blob_batch` in the embedding
loop (lines 101, 136, 142) but **never read**. The candidate IDs are already
tracked via `all_candidate_ids`. Remove it and its references.

**Done when:** `grep id_batch precompute.py` returns nothing and the script
still runs end-to-end.

---

## 2. 🐳 Add a `.dockerignore`

**Difficulty:** trivial · **Files:** `.dockerignore` (new)

`Dockerfile` does `COPY . .`, which pulls the multi-GB `venv/`, the ~487 MB
`data/` directory, `artifacts/`, `.git/`, and `output/` into the build context.
Add a `.dockerignore` excluding all of these. Data should be volume-mounted
at runtime anyway.

**Done when:** `docker build` context is a few MB instead of several GB.

---

## 3. ⏱️ Stale performance numbers in README

**Difficulty:** trivial · **Files:** `README.md`

The Performance table still says "Ranking & Validation ~1.5 min". After the
byte-offset index and batch-loading fixes, ranking completes in roughly 5
seconds. Re-measure with `time python rank.py` and update the table.

**Done when:** the Performance section matches a fresh local run.

---

## 4. 🏛️ Unreachable keywords in `TIER_2_INSTITUTIONS`

**Difficulty:** easy · **Files:** `config.py`, `signals.py`

`TIER_1_INSTITUTIONS` contains `"iiit"` (line 123) and is checked first
(lines 332–334), so the `"iiit"` entry in `TIER_2_INSTITUTIONS` can never
match. Same for `"nit "` (tier-1, with trailing space) vs `"nit"` (tier-2,
no space — which would also false-match words like *monitoring*).

Deduplicate the lists and add word-boundary matching in
`_check_education_tier`'s keyword fallback.

**Done when:** no keyword appears in both lists and `"monitoring university"`
doesn't classify as tier-2.

---

## 5. 💥 Crash-proof `build_outreach_pack` name handling

**Difficulty:** easy · **Files:** `app.py`

Line 646: `name.split()[0] if name else "there"` raises `IndexError` when
`anonymized_name` is whitespace-only (e.g. `"  "` — `"  ".split()` returns
`[]`, so `[0]` fails). Use `name.split()[0] if name.split() else "there"`
or equivalent.

**Done when:** a candidate dict with `anonymized_name: "  "` produces a
greeting of `"there"` instead of crashing the Export tab.

---

## 6. 🛡️ Defend against `null` role descriptions in `compute_technical_fit`

**Difficulty:** easy · **Files:** `signals.py`

`compute_technical_fit` builds its text blob using `r.get("description", "")`
(line 91), but Python's `dict.get` returns `None` (not `""`) when the JSON
value is literally `null`. This propagates into `" ".join(...)` and crashes.
Other call sites already use `(r.get("description", "") or "")` — apply the
same guard here and to the `headline`/`summary`/`current_title` concatenation.

**Done when:** a candidate with `"description": null` scores without crashing.

---

## 7. 📦 Demographics cache never invalidates

**Difficulty:** easy · **Files:** `app.py`

`pool_demographics()` (line 128) writes `artifacts/demographics.csv` once
and returns it forever, even if `data/candidates.jsonl` changes. Compare the
mtime of the cache file against the JSONL and rebuild when stale.

**Done when:** touching `candidates.jsonl` causes a rebuild on next dashboard
start.

---

## 8. 🧪 Add a pytest regression suite

**Difficulty:** medium · **Files:** `tests/` (new), `requirements.txt`

All scoring edge cases are currently verified ad hoc. Create:

| Test file | What it covers |
|---|---|
| `tests/test_signals.py` | Technical fit, career quality, availability, seniority |
| `tests/test_disqualify.py` | Honeypot, ghost, pure research, consulting/no-code/CV penalties |
| `tests/test_evidence.py` | Evidence collection, reasoning generation |
| `tests/test_engine.py` | MMR diversity, stability analysis, top-k determinism |
| `tests/test_textmatch.py` | Word-boundary patterns, signal scanning |

Specific scenarios to cover:
- `"version control"` does **not** trigger the CV/robotics penalty
- all-empty company names do **not** zero `career_quality`
- `"Senior Principal Engineer"` → seniority level 4; `"International Sales Lead"` → 3
- newest-first career history still detects upward progression
- tier-1/tier-2 education institution strings parse correctly
- `generate_reasoning` only claims production evidence when present
- `mmr_rerank` returns unique indices and starts with the top-scored candidate
- `top_k_indices` ordering is deterministic on ties

**Done when:** `pytest` passes and runs in under 30 s without artifacts.

---

## 9. 📋 Export CSV tie-breaking doesn't match the challenge spec

**Difficulty:** medium · **Files:** `engine.py`, `app.py`

The spec requires ties (at 3 displayed decimals) to be broken by ascending
`candidate_id`. `rank.py` does this correctly (`round(x[0], 3)` + `x[1]`
secondary sort), but:

- `engine.top_k_indices` (used by the dashboard) rounds to **6 decimals**
  and tie-breaks by **array index** (line 42)
- `build_submission_csv` inherits that ordering

Align the dashboard export path with the spec: round to 3 decimals and
tie-break by `candidate_id` ascending.

**Done when:** the dashboard-exported CSV passes
`python data/validate_submission.py <file>` including tie-break ordering for
synthetic tied scores.

---

## 10. 📝 `app.py` still says "RecruiterIQ" in page title

**Difficulty:** trivial · **Files:** `app.py`

The project was renamed from RecruiterIQ to SignalHire. The page title
(line 38) and sidebar heading (line 225) still display the old name, plus
the module docstring (line 1) and title (line 719).

**Done when:** every visible "RecruiterIQ" string in `app.py` reads
"SignalHire".