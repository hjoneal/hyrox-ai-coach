# Project Log

A running record of decisions, rationale, and results — written as input to the final project write-up. Each entry: what was decided/found, why, alternatives considered, and evidence where applicable.

---

## 2026-06-04 — Project repositioning: ML portfolio, not LLM agent

**Decision:** Reposition the project as a data science portfolio piece. Drop the planned LangGraph/LLM coaching agent and ChromaDB entirely (never implemented; placeholder files removed).

**Context:** The project began as an ML engineering exercise (in-race finish time prediction), pivoted toward an LLM coaching agent for AI-engineer interviews, and is now being refocused for data science roles. The LLM agent demonstrated orchestration, not modeling — its analytical core was percentile lookups, which would not withstand DS interview scrutiny.

**Alternatives considered:**
- *Keep the agent as a thin demo layer* — rejected for now; split focus dilutes the modeling narrative. Could be revisited as a presentation layer once the models exist.
- *Maintain parallel AI-engineer track* — rejected; maintenance cost with no DS interview value.

**The pivot itself is a talking point:** scoping a project to its audience, and recognizing that an LLM wrapper around a lookup table is not a modeling contribution.

## 2026-06-04 — Core analyses selected

**Decision:** Three connected analyses, chained so that the outputs of (2) and (3) become candidate features for (1):

1. **In-race finish time prediction (flagship).** One model per race checkpoint (run 1 → station 8), using only information available at that checkpoint. Emphases: leakage discipline, domain-naive baseline (proportional extrapolation), leave-event-out cross-validation, and calibrated uncertainty (quantile regression / conformal intervals). Headline figure: prediction interval width vs. race progress.
2. **Venue difficulty model.** Hierarchical/mixed-effects model with venue random effects on (log) finish time. Known identification challenge: venue effects are confounded with field strength; mitigation via field-composition covariates and possibly within-athlete contrasts from repeat competitors (light entity resolution).
3. **Pacing archetype clustering.** GMM over splits normalized as fraction of total time; k selected by BIC. Includes an explicit confounding analysis (does pacing style predict performance *within* finish-time bands, or does fitness drive both?).

**Alternatives considered:**
- *Athlete progression via entity resolution* — declined as a core piece (match-quality risk, less visual payoff); may be used in light form for venue-model identification.
- *Race plan optimizer* — declined; product-shaped, thin on technique.

**Scope rule:** three analyses is the ceiling. If timeline pressure hits, cut clustering first (most commoditized technique of the three).

**Dataset decision:** flagship trains on the OPEN division singles dataset (~20.9k athletes, 24 events, seasons 5–6). Pro Doubles data is team-level (shared station splits) and unsuitable for individual modeling; the full doubles scrape is off the critical path. The December race is motivation/context, not a deliverable.

## 2026-06-04 — Phase 0: repo realignment

**Actions:** removed empty placeholder modules (`src/agent/`, `app/`, `src/models/predictor.py`, boilerplate `main.py`), the obsolete LangGraph plan doc, and a stale `requirements.txt` that listed unused dependencies (langchain, chromadb, openai, streamlit). `pyproject.toml` + `uv.lock` are now the single source of dependency truth. Added modeling/EDA dependencies (scikit-learn, lightgbm, matplotlib, seaborn). README rewritten around the DS narrative.

**Addendum:** repo renamed `hyrox-ai-coach` → `hyrox-performance-analytics` (GitHub + local) to match the repositioning; hardcoded absolute paths in `scripts/` replaced with `__file__`-relative paths. Known issue found during verification: `scripts/test_features.py` imports `parse_timedelta_string`, which no longer exists in `features.py` — pre-existing staleness, to be fixed in the Phase 1 audit.

**Next:** Phase 1 — data audit + EDA. The audit determines downstream feasibility (gender composition of the OPEN scrape, missing/zero splits, DNF handling, duplicates, venue count adequacy for the hierarchical model).

## 2026-06-04 — Phase 1: data audit findings

Full audit in [report notebook 1](../notebooks/Harry_Neal_Hyrox_1_data_acquisition_cleaning_eda.ipynb) (originally a standalone audit notebook, folded into the report series on 2026-06-05). Answers to the five audit questions:

1. **Gender was never captured.** The original scrape ran without a sex filter, and the detail-page parser silently failed on `age_group`, `nationality`, `division`, and `disqual_reason` (all-NaN in every row). Men's and women's races were mixed and unlabeled — a serious problem for every planned model (finish-time distributions are offset by gender, and the venue model's field-composition covariate depends on it). See recovery decision below.
2. **Zero splits encode missing data.** No NaNs exist in the splits; zeros are missed checkpoint mats (142 rows post-dedup, ~0.7%, concentrated in late runs — consistent with mat misses, not data entry noise).
3. **The source lists finishers only.** Every row has a positive overall time and no disqualification marker. No DNF cleaning is needed, but all models are *conditional on finishing*; no survivorship correction is possible from this data.
4. **Duplicates have two mechanisms.** (a) 455 exact duplicate rows: the results site renders duplicate `<li>` rows per athlete (responsive layouts) and the scraper followed both — render artifacts, dropped. (b) Repeated bib numbers within an event are *different people* (bibs reused across waves) — `bib_number` is not a key. Same-name pairs within an event (45 groups) are distinct athletes; names are not IDs either.
5. **Venue coverage is adequate for the hierarchical model.** 20 venues / 24 events, 226–2,828 athletes per event, 4 venues repeated across seasons (helps separate venue from event-edition effects). Raw venue medians span ~24 min (Barcelona 82.6 → Singapore 106.8), but gender mix explains little of it (r = −0.19), so the Phase 3 confounder is field strength broadly, not just M/W ratio.

**Event-level anomaly:** S6 Paris published no roxzone times at all (713 rows). The gap `overall − Σsplits` matches the global roxzone distribution (median 418s vs 399s), so roxzone is imputed as that gap. Similarly, `total_run`/`total_stations`/`best_run_lap` were unpublished (zero) for ~720 rows and are recomputed from splits.

**Side finding:** `results.hyrox.com` now returns 403 to the default python-requests User-Agent; the scraper session sets browser-like headers (`HyroxScraper.REQUEST_HEADERS`). The stale `scripts/test_features.py` (imported a function deleted from `features.py`, read an obsolete-schema test CSV) was rewritten against the current schema.

## 2026-06-04 — Gender recovery via list-page label scrape

**Decision:** recover gender (plus age group and nationality) with a *list-page-only* scrape, sex-filtered, joined back on `(event_id, name, finish_time)`.

**Why this option:** the sex-filtered list pages carry name, finish time, age group, and nationality at ~100 athletes per request — ~430 requests for all 24 events vs ~21k detail-page requests for a full re-scrape. Proceeding gender-blind was rejected: an unlabeled mixture inflates variance in the flagship model and removes the venue model's field-composition covariate.

**Result:** 20,791 labels scraped (`scripts/scrape_gender_labels.py`, cached per event×sex under `data/raw/labels/`); exact join labeled **20,429 / 20,432 rows (99.99%)** — 3 unmatched, zero fallback matches needed. Composition: 69% men (14,109), 31% women (6,320). The join key was validated before the full run (list finish time == detail overall time exactly).

**Joint payoff:** age group (a covariate the detail scrape lost) and nationality came back for free in the same pass.

## 2026-06-04 — Cleaning policy and clean modeling table

**Decision:** every audit finding is implemented as an explicit, counted rule in [`src/processing/cleaning.py`](../src/processing/cleaning.py) (R1–R6); `scripts/build_clean_dataset.py` writes `data/processed/hyrox_clean.csv` and prints the audit trail on every build. Rows are **flagged, not dropped** (`is_modeling_row`), so each analysis chooses its own strictness — only exact duplicates are physically removed.

| rule | action | count |
|------|--------|-------|
| R1 | drop exact duplicate rows | 455 |
| R2 | impute missing roxzone as `overall − Σsplits`; recompute derived totals | 699 imputed |
| R3 | zero splits → NaN + `n_missing_splits` | 142 rows |
| R4 | `splits_consistent` flag (±60 s reconciliation) | 229 fail |
| R5 | gender/age/nationality label join | 99.99% labeled |
| R6 | drop dead (all-NaN / all-zero) columns | 6 cols |

**Retention: 20,203 modeling rows of 20,887 scraped (96.7%).**

**Known caveat:** rows with imputed roxzone (R2) reconcile in R4 by construction — the consistency check only bites on rows that arrived with a published roxzone. Documented in the notebook.

**Alternatives considered:** hard-dropping flagged rows (rejected — different analyses tolerate different incompleteness; e.g. the clustering can use rows the checkpoint models can't); imputing zero splits from neighboring segments (rejected for now — 0.7% of rows isn't worth the modeling risk, revisit if checkpoint models need the late-run rows).

**Next:** Phase 2 — in-race prediction. Per-checkpoint feature sets from `is_modeling_row` data with gender as a covariate, proportional-extrapolation baseline, leave-event-out CV.

## 2026-06-05 — Project report as a notebook series

**Decision:** the project report is a continuously-updated notebook series in `notebooks/`, structured and written in the style of the Spotify MoodGrid report notebooks (Introduction → Method (acquisition/cleaning/EDA) → Modelling and Results → Discussion, with charts throughout). Planned series:

1. `Harry_Neal_Hyrox_1_data_acquisition_cleaning_eda.ipynb` — written now, subsumes the Phase 1 audit
2. In-race finish time prediction (Phase 2)
3. Venue difficulty modelling (Phase 3)
4. Pacing archetypes + overall discussion (Phases 4–5)

**Alternatives considered:** a single growing report notebook (rejected — three analyses would make it unreadably long); a hybrid headline-notebook linking to working notebooks (rejected — duplication, two sources of truth). The standalone `01_data_audit.ipynb` was folded into report notebook 1 and deleted for the same single-source-of-truth reason.
