# LangGraph Agent Implementation Plan

## Overview

The Hyrox AI Coach agent is a conversational ReAct-style agent built with LangGraph. It uses Claude (via `langchain-anthropic`) as its LLM backbone and exposes 5 tools that allow it to load athlete data, analyze performance against the dataset, identify weaknesses, and generate training recommendations.

## Architecture

ReAct StateGraph: `llm_node` ↔ `tool_node` loop with conditional edges.

```
      [START]
         |
         v
    +----------+
    |   llm    | <---+
    +----------+     |
         |           |
    (tool_calls?)    |
    /          \     |
  yes          no    |
   |            |    |
   v            v    |
+------+      [END]  |
|tools |             |
+------+ -----------+
```

## Dependencies to Add

```toml
dependencies = [
    # ... existing deps ...
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "anthropic>=0.40.0",
]
```

Then run `~/.local/bin/uv sync`.

## File Structure

```
src/agent/
    __init__.py          # Already exists (empty) - add exports
    coach.py             # Already exists (empty) - main agent graph
    state.py             # NEW: State schema (TypedDict for graph state)
    tools.py             # NEW: All 5 tool implementations
    prompts.py           # NEW: System prompt and prompt templates
    data_loader.py       # NEW: Data loading utilities (CSV loading, percentile computation)
scripts/
    run_coach.py         # NEW: CLI entry point for interactive chat
```

Total: 4 new files + 2 files to modify (`coach.py`, `__init__.py`, `pyproject.toml`).

## State Schema (`src/agent/state.py`)

```python
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AthleteData(TypedDict, total=False):
    """Athlete performance data loaded from URL or manual entry."""
    name: str
    overall_time: int           # seconds
    run_1: int  # through run_8
    station_1: int  # through station_8 (SkiErg, Sled Push, Sled Pull, Burpees, Row, Farmers, Lunges, Wall Balls)
    total_run: int
    total_stations: int
    roxzone_time: int
    rank_overall: int
    age_group: str
    nationality: str
    event_name: str
    division: str
    source: str  # "url" or "manual"


class CoachState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    athlete_data: Optional[AthleteData]
```

Key decisions:
- `messages` uses LangGraph's `add_messages` reducer (append, not overwrite)
- `athlete_data` is `Optional` — tools like `compute_percentile_rankings` require it but general chat doesn't

## Data Loading Utilities (`src/agent/data_loader.py`)

`ReferenceDataset` singleton that lazy-loads `data/processed/hyrox_features.csv` (~20,887 rows).

Two main methods:
- **`compute_percentiles(athlete_splits)`** — For each of the 16 splits + overall/totals, uses `scipy.stats.percentileofscore()`. Inverts since lower time = better: `percentile = 100 - raw_percentile`.
- **`identify_weaknesses(athlete_splits)`** — Computes all percentiles, then `gap = segment_percentile - overall_percentile`. Negative gaps = relative weaknesses. Sorted by severity.

## Tool Implementations (`src/agent/tools.py`)

### Tool 1: `load_athlete_from_url`
- Parses season number from URL path
- Uses existing `HyroxScraper.parse_participant_details(url)`
- Stores parsed data in module-level `_current_athlete` singleton
- Returns formatted summary string

### Tool 2: `load_athlete_from_manual_splits`
- 18+ params: name, overall_time, run_1..run_8, station_1..station_8, roxzone_time
- All times in MM:SS or H:MM:SS format
- Uses existing `parse_time_to_seconds()` from `src/ingestion/scraper.py`
- Computes `total_run` and `total_stations` as sums

### Tool 3: `compute_percentile_rankings`
- Requires athlete loaded first
- Calls `ReferenceDataset.get().compute_percentiles()`
- Returns formatted table with percentile for each segment

### Tool 4: `identify_weak_stations`
- Requires athlete loaded first
- Calls `ReferenceDataset.get().identify_weaknesses()`
- Categorizes by severity: significant (>15pts below), moderate (5-15pts), strengths
- Uses station groupings from `StationFeatureExtractor`: CARDIO_STATIONS, STRENGTH_STATIONS, ENDURANCE_STATIONS

### Tool 5: `generate_training_recommendations`
- Requires athlete loaded first
- Pre-defined `STATION_RECOMMENDATIONS` mapping per station (specific drills, exercises)
- Also includes pacing advice based on run_acceleration, station_fatigue_index, roxzone_ratio
- Outputs prioritized training plan with weekly template

### State Management
Module-level `_current_athlete` singleton — fine for single-user CLI. For future multi-user web deployment, refactor to session-scoped storage.

## Agent Graph (`src/agent/coach.py`)

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import ToolNode

def build_graph() -> StateGraph:
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3, max_tokens=4096)
    llm_with_tools = llm.bind_tools(TOOLS)

    # llm_node: prepends system message, invokes LLM
    # tool_node: ToolNode(TOOLS) from langgraph.prebuilt

    graph = StateGraph(CoachState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile()
```

## System Prompt Design

Covers:
- Hyrox race format (8 rounds, station details, weights)
- Station categories (Cardio/Strength/Endurance)
- Typical finish times by level
- Tool workflow guidance (load → percentiles → weaknesses → recommendations)
- Formatting guidelines (MM:SS, tables, encourage but be honest)

## CLI Runner (`scripts/run_coach.py`)

Interactive chat loop:
- Welcome banner with capabilities
- `while True` input loop
- Invokes `agent.invoke({"messages": messages})`
- Prints last AI message
- Requires `ANTHROPIC_API_KEY` env var

## Implementation Phases

### Phase 1: Foundation
1. Create `src/agent/state.py` — CoachState + AthleteData
2. Create `src/agent/data_loader.py` — ReferenceDataset with percentile/weakness logic
3. Test data loading independently

### Phase 2: Tools
4. Create `src/agent/tools.py` — all 5 tools
5. `load_athlete_from_url` using `HyroxScraper.parse_participant_details()`
6. `load_athlete_from_manual_splits` with `parse_time_to_seconds()`
7. `compute_percentile_rankings` using ReferenceDataset
8. `identify_weak_stations` using weakness analysis
9. `generate_training_recommendations` with recommendation mappings

### Phase 3: Agent Graph
10. Create `src/agent/prompts.py`
11. Implement `src/agent/coach.py` with `build_graph()`

### Phase 4: CLI + Testing
12. Create `scripts/run_coach.py`
13. End-to-end test: general chat → load from URL → percentiles → weaknesses → recommendations
14. Update `pyproject.toml` dependencies

## Key Existing Code to Reuse

| Reference | Location | Usage |
|-----------|----------|-------|
| `parse_time_to_seconds()` | `src/ingestion/scraper.py:28-52` | Parse MM:SS strings in manual splits |
| `HyroxScraper.parse_participant_details()` | `src/ingestion/scraper.py:122-227` | Load athlete from URL |
| `StationFeatureExtractor.STATION_NAMES` | `src/processing/features.py:89-98` | Human-readable station names |
| `CARDIO/STRENGTH/ENDURANCE_STATIONS` | `src/processing/features.py:101-103` | Station categorization |
| `data/processed/hyrox_features.csv` | ~20,887 rows, 71 cols | Reference dataset for percentiles |

## Environment

- Requires `ANTHROPIC_API_KEY` environment variable
- `langchain-anthropic` reads it automatically
