# Hyrox Performance Analytics

Data-driven analysis of [Hyrox](https://hyrox.com/) race results, built on a self-collected dataset of ~21,000 athlete performances scraped from official results. The project centres on three connected analyses:

1. **In-race finish time prediction** — checkpoint-by-checkpoint models with calibrated uncertainty: "given your splits so far, when will you finish?"
2. **Venue difficulty modeling** — hierarchical models quantifying how much course/venue conditions (sled carpet friction, layout) shift finish times
3. **Pacing archetype clustering** — unsupervised discovery of pacing strategies, and whether strategy predicts performance independent of fitness

The analyses chain together: venue effects and pacing-cluster membership are candidate features for the prediction model, so the project reads as one modeling story rather than three demos.

Decisions, methodology, and results are logged as they happen in [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

## What is Hyrox?

Hyrox is a global fitness racing series combining **8 x 1km runs** with **8 functional workout stations**, performed in a fixed order:

| Station | Exercise | Description |
|---------|----------|-------------|
| 1 | SkiErg | 1000m on ski ergometer |
| 2 | Sled Push | 50m pushing weighted sled |
| 3 | Sled Pull | 50m pulling weighted sled |
| 4 | Burpee Broad Jumps | 80m of burpee broad jumps |
| 5 | Rowing | 1000m on rowing ergometer |
| 6 | Farmers Carry | 200m carrying kettlebells |
| 7 | Sandbag Lunges | 100m walking lunges with sandbag |
| 8 | Wall Balls | 100 wall ball repetitions |

Typical finish times range from **~55 minutes** (elite) to **2+ hours**. Each performance yields 17+ timed segments (8 runs, 8 stations, transition "Roxzone" time), making races unusually rich, structured time-series-like observations.

## Why this dataset is interesting

- **Sequential structure**: every athlete passes the same 16 checkpoints in order — natural ground for in-race prediction and leakage-discipline questions ("what is knowable at checkpoint k?")
- **Grouped structure**: athletes nest within events/venues — random splits leak event effects, motivating leave-event-out validation and hierarchical modeling
- **Real-world mess**: scraped HTML, missing/zero splits, DNFs, duplicate entries, venue-to-venue field strength differences — every cleaning and identification decision has to be made explicitly (and is documented in the project log)

## Project Status

| Component | Status |
|-----------|--------|
| Web scraper (official results, seasons 5–8, division/gender aware) | **Done** |
| Combined OPEN dataset (~20.9k athletes, 24 events, S5–S6) | **Done** |
| Feature engineering pipeline (run / station / pacing features) | **Done** |
| Data audit + EDA | In progress |
| In-race finish time prediction (flagship) | Planned |
| Venue difficulty model | Planned |
| Pacing archetype clustering | Planned |

## Project Structure

```
hyrox-performance-analytics/
├── src/
│   ├── ingestion/
│   │   ├── scraper.py      # Scraper for results.hyrox.com (division/gender aware)
│   │   └── events.py       # Event registry (seasons 5-8), Division/Gender enums
│   ├── processing/
│   │   └── features.py     # Feature engineering (run, station, pacing extractors)
│   └── models/             # Modeling code (Phase 2+)
├── scripts/
│   ├── run_full_scrape.py  # Full OPEN dataset collection
│   ├── test_scraper.py     # Scraper validation
│   └── test_features.py    # Feature pipeline validation
├── notebooks/              # EDA and analysis notebooks
├── docs/
│   └── PROJECT_LOG.md      # Running log of decisions and results
├── data/                   # (gitignored) raw scrapes and processed features
└── pyproject.toml
```

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/hjoneal/hyrox-performance-analytics.git
cd hyrox-performance-analytics
uv sync
```

## Usage

### Scraping data

```python
from src.ingestion.scraper import HyroxScraper
from src.ingestion.events import ALL_EVENTS

scraper = HyroxScraper(season=5)
df = scraper.scrape_multiple_events(
    events=ALL_EVENTS,
    output_dir="data/raw/events",
    save_intermediate=True,   # per-event CSV cache, resumable
)
```

### Feature engineering

```python
import pandas as pd
from src.processing.features import HyroxFeatureEngineer

df = pd.read_csv("data/raw/hyrox_combined.csv")
engineer = HyroxFeatureEngineer()
df_features = engineer.fit_transform(df)
```

Engineered features cover run pacing (mean, variability, fatigue trend), station profiles (cardio/strength/endurance splits, fatigue index), and race-level pacing (positive/negative split, finish strength). See `src/processing/features.py`.

## Roadmap

- **Phase 1 — Data audit + EDA**: missing/zero splits, DNF policy, duplicates, field composition per event; documented cleaning policy and clean modeling table
- **Phase 2 — In-race prediction**: per-checkpoint feature sets, naive extrapolation baseline, gradient-boosted models, leave-event-out CV, quantile/conformal prediction intervals, calibration analysis
- **Phase 3 — Venue difficulty**: mixed-effects / Bayesian hierarchical model with venue random effects; field-strength confounding addressed explicitly
- **Phase 4 — Pacing archetypes**: GMM over normalized split profiles, model selection via BIC, performance analysis stratified by finish-time band
- **Phase 5 — Write-up**: headline figures, per-analysis summaries, limitations

## Project history

This repo started as a finish-time prediction exercise, briefly detoured toward an LLM coaching agent, and was deliberately re-scoped to the modeling work above — the agent added orchestration, not analysis. The detour is preserved in git history and `docs/PROJECT_LOG.md`.

## Data Sources

All data scraped from the official results site (`results.hyrox.com`), seasons 5–8. The current modeling dataset covers 24 OPEN-division events across seasons 5–6 (~20.9k athletes). Raw data is not committed to the repo.

## License

MIT License - see [LICENSE](LICENSE) for details.
