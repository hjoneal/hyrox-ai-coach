# Hyrox AI Coach

An AI-powered performance analysis and coaching system for [Hyrox](https://hyrox.com/) athletes. This project scrapes competition data, engineers predictive features, and (planned) provides personalised training insights using machine learning and LLM-based coaching.

## What is Hyrox?

Hyrox is a global fitness racing series combining **8 x 1km runs** with **8 functional workout stations**:

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

Typical finish times range from **45 minutes** (elite) to **2+ hours** (beginners).

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| Data Scraping | **Working** | Scrapes participant splits from official results |
| Feature Engineering | **Working** | 34 engineered features for ML modeling |
| ML Models | Planned | Finish time prediction models |
| AI Coach Agent | Planned | LLM-powered personalised coaching |
| Web Dashboard | Planned | Streamlit interface for insights |

### Current Progress

- [x] Web scraper for Hyrox results (seasons 5-8)
- [x] Batch scraping with resumability
- [x] Feature engineering pipeline (run, station, pacing features)
- [x] Feature validation and correlation analysis
- [ ] Expanded dataset (25 events, ~8000 participants)
- [ ] Predictive modeling (XGBoost, scikit-learn)
- [ ] AI coaching agent (LangChain + GPT)
- [ ] Web dashboard (Streamlit)

## Project Structure

```
hyrox-ai-coach/
├── src/
│   ├── ingestion/
│   │   ├── scraper.py      # Web scraper for Hyrox results
│   │   └── events.py       # Event configuration (25 curated events)
│   ├── processing/
│   │   └── features.py     # Feature engineering pipeline
│   ├── models/
│   │   └── predictor.py    # ML models (planned)
│   └── agent/
│       └── coach.py        # AI coaching agent (planned)
├── app/
│   └── main.py             # Streamlit web app (planned)
├── scripts/
│   ├── test_scraper.py     # Scraper validation
│   ├── test_features.py    # Feature pipeline validation
│   └── run_full_scrape.py  # Full dataset collection
├── data/
│   ├── raw/                # Scraped CSV files
│   └── processed/          # Engineered features
├── notebooks/
│   └── hyrox-data-scraping.ipynb  # Exploration notebook
├── pyproject.toml
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hyrox-ai-coach.git
cd hyrox-ai-coach

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

## Usage

### Scraping Data

```python
from src.ingestion.scraper import HyroxScraper
from src.ingestion.events import get_events

# Initialise scraper
scraper = HyroxScraper(season=5)

# Scrape multiple events
events = get_events(seasons=[5, 6], limit=10)
df = scraper.scrape_multiple_events(events, output_dir="data/raw/events")
```

### Feature Engineering

```python
from src.processing.features import HyroxFeatureEngineer
import pandas as pd

# Load scraped data
df = pd.read_csv("data/raw/hyrox_combined.csv")

# Generate features
engineer = HyroxFeatureEngineer()
df_features = engineer.fit_transform(df)

# Get target variable
target = engineer.get_target(df_features)
print(f"Features: {len(engineer.get_feature_names())}")
```

### Running Tests

```bash
# Test scraper with 2 events
python scripts/test_scraper.py

# Test feature pipeline
python scripts/test_features.py
```

## Engineered Features

### Run Features (10)
| Feature | Description |
|---------|-------------|
| `run_mean_seconds` | Average run split time |
| `run_std_seconds` | Run time consistency (std dev) |
| `run_cv` | Coefficient of variation |
| `run_acceleration` | Second half vs first half pacing |
| `run_trend_slope` | Fatigue trend across 8 runs |
| `run_slowest_idx` | Which run was slowest (1-8) |
| `run_fastest_idx` | Which run was fastest (1-8) |
| `run_range` | Difference between slowest and fastest |

### Station Features (12)
| Feature | Description |
|---------|-------------|
| `station_mean_seconds` | Average station time |
| `cardio_time` | SkiErg + Row combined time |
| `strength_time` | Sled Push + Pull + Farmers time |
| `endurance_time` | Burpees + Lunges + Wall Balls time |
| `cardio_ratio` | Cardio proportion of total station time |
| `strength_ratio` | Strength proportion |
| `endurance_ratio` | Endurance proportion |
| `station_fatigue_index` | Stations 5-8 avg / Stations 1-4 avg |

### Pacing Features (8)
| Feature | Description |
|---------|-------------|
| `run_station_ratio` | Running vs station time balance |
| `positive_split` | 1 if second half slower than first |
| `split_difference` | Time difference between halves |
| `finish_strength` | Last 2 runs vs first 2 runs |

## Roadmap

### Phase 1: Data Foundation (Current)
- [x] Build web scraper for Hyrox results
- [x] Create event configuration for 25 events (seasons 5-6)
- [x] Implement feature engineering pipeline
- [ ] Collect full dataset (~8000 participants)
- [ ] Data quality validation and cleaning

### Phase 2: Predictive Modeling
- [ ] Exploratory data analysis
- [ ] Baseline regression model (Linear Regression)
- [ ] Advanced models (XGBoost, Random Forest)
- [ ] Feature importance analysis
- [ ] Model evaluation and selection
- [ ] Hyperparameter tuning

### Phase 3: AI Coaching Agent
- [ ] Design coaching prompt templates
- [ ] Implement LangChain agent with tools
- [ ] Create performance analysis functions
- [ ] Build weakness identification logic
- [ ] Training recommendation generation
- [ ] Conversation memory with ChromaDB

### Phase 4: Web Application
- [ ] Streamlit dashboard layout
- [ ] Performance visualisation charts
- [ ] AI coach chat interface
- [ ] Race simulation/prediction tool
- [ ] Training plan generator

## Data Sources

Data is scraped from the official Hyrox results website:
- `results.hyrox.com` (Seasons 5-8)

### Available Events (25 curated)

**Season 5 (15 events):** New York, Leipzig, Birmingham, Valencia, Amsterdam, Chicago, Berlin, London, Los Angeles, Dallas, Hamburg, Glasgow, Manchester, Stockholm, Barcelona

**Season 6 (10 events):** New York, London, Paris, Birmingham, Amsterdam, Munich, Frankfurt, Sydney, Melbourne, Singapore

## Technical Stack

| Category | Technologies |
|----------|-------------|
| **Data Collection** | BeautifulSoup, Requests |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **AI/LLM** | LangChain, LangGraph, OpenAI |
| **Vector Store** | ChromaDB |
| **Web App** | Streamlit |
| **Package Management** | uv |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Hyrox](https://hyrox.com/) for creating the fitness racing format
- Competition data sourced from official Hyrox results
