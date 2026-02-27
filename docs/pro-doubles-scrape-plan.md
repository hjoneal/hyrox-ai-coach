# Pro Doubles Scrape Plan

## Discovery Results

Ran `scripts/discover_events.py` on 2026-02-27. Probed all 34 events across seasons 5-8 for PRO_DOUBLES (HDP) division, MEN gender filter.

**Result: 32/34 events have Pro Doubles MEN data.**

### Events with data (32)

#### Season 5 (14/15 events)
| Event | ID | Status |
|-------|-----|--------|
| S5 New York 2022 | 2EFMS4JI321 | Found 100+ |
| S5 Leipzig 2022 | 2EFMS4JI35D | Found 100+ |
| S5 Birmingham 2022 | — | **Timed out** |
| S5 Valencia 2022 | 2EFMS4JI371 | Found 100+ |
| S5 Amsterdam 2022 | 2EFMS4JI385 | Found 100+ |
| S5 Chicago 2022 | JGDMS4JI39A | Found 100+ |
| S5 Berlin 2022 | 2EFMS4JI399 | Found 100+ |
| S5 London 2022 | 2EFMS4JI3AD | Found 100+ |
| S5 Los Angeles 2022 | JGDMS4JI3FE | Found 100+ |
| S5 Dallas 2022 | JGDMS4JI3D7 | Found 100+ |
| S5 Hamburg 2022 | JGDMS4JI3E9 | Found 100+ |
| S5 Glasgow 2023 | JGDMS4JI439 | Found 100+ |
| S5 Manchester 2023 | JGDMS4JI425 | Found 100+ |
| S5 Stockholm 2023 | JGDMS4JI467 | Found 100+ |
| S5 Barcelona 2023 | JGDMS4JI466 | Found 100+ |

#### Season 6 (9/10 events)
| Event | ID | Status |
|-------|-----|--------|
| S6 New York 2023 | — | **Timed out** |
| S6 London 2023 | JGDMS4JI62E | Found 100+ |
| S6 Paris 2023 | JGDMS4JI579 | Found 100+ |
| S6 Birmingham 2023 | JGDMS4JI5E2 | Found 100+ |
| S6 Amsterdam 2023 | JGDMS4JI606 | Found 100+ |
| S6 Munich 2023 | JGDMS4JI5C9 | Found 100+ |
| S6 Frankfurt 2023 | JGDMS4JI655 | Found 100+ |
| S6 Sydney 2023 | JGDMS4JI515 | Found 100+ |
| S6 Melbourne 2023 | JGDMS4JI516 | Found 100+ |
| S6 Singapore 2023 | JGDMS4JI58D | Found 100+ |

#### Season 7 (5/5 events)
| Event | ID | Status |
|-------|-----|--------|
| S7 Stockholm 2024 | LR3MS4JI11FA | Found 34 |
| S7 Rimini 2024 | LR3MS4JI1213 | Found 100+ |
| S7 Malaga 2024 | LR3MS4JI126C | Found 100+ |
| S7 Manchester 2024 | LR3MS4JI12C5 | Found 100+ |
| S7 Cologne 2024 | LR3MS4JI131E | Found 100+ |

#### Season 8 (4/4 events)
| Event | ID | Status |
|-------|-----|--------|
| S8 Amsterdam 2024 | LR3MS4JI1377 | Found 100+ |
| S8 London 2024 | LR3MS4JI13D0 | Found 100+ |
| S8 Hamburg 2024 | LR3MS4JI1429 | Found 100+ |
| S8 Rome 2025 | LR3MS4JI1482 | Found 100+ |

### Timed-out events (2)
- S5 Birmingham 2022 — may have data, just timed out
- S6 New York 2023 — may have data, just timed out

These can be retried individually.

## Test Scraper Verification

Ran `scripts/test_doubles_scraper.py` with 2 events (S5 New York 2022, S5 Leipzig 2022).

- **S5 New York 2022**: Loaded from cache, 693 participants
- **S5 Leipzig 2022**: Scraping live, reached 4125+ participants before being stopped (very large event)
- Scraper is working correctly with division prefix (`HDP_`) and gender filter (`&sex=M`)

## Next Steps

1. **Run full scrape**: `scripts/run_doubles_scrape.py` — scrapes all 32 valid events
   - Output: `data/raw/hyrox_mens_pro_doubles_combined.csv`
   - Features: `data/processed/hyrox_mens_pro_doubles_features.csv`
2. **Retry timed-out events**: S5 Birmingham 2022, S6 New York 2023
3. Consider scraping PRO_DOUBLES WOMEN as well

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/discover_events.py` | Probe events for pro doubles data |
| `scripts/test_doubles_scraper.py` | Test with 2 events, validate structure |
| `scripts/run_doubles_scrape.py` | Full scrape of all events |
