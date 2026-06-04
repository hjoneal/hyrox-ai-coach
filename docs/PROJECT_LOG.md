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

**Next:** Phase 1 — data audit + EDA. The audit determines downstream feasibility (gender composition of the OPEN scrape, missing/zero splits, DNF handling, duplicates, venue count adequacy for the hierarchical model).
