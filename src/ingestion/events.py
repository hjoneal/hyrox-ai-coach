"""
Event configuration for Hyrox data scraping.

Contains curated event IDs from seasons 5-6 for Open division scraping.
"""

from dataclasses import dataclass
from typing import List
from enum import Enum


class Division(Enum):
    """Hyrox competition divisions."""
    OPEN = "H"
    PRO = "HPRO"
    ELITE = "HE"
    DOUBLES = "HD"
    RELAY = "HMR"
    GORUCK = "HG"
    GORUCK_DOUBLES = "HDG"


@dataclass
class HyroxEventConfig:
    """Configuration for a single Hyrox event."""
    event_id: str
    season: int
    name: str
    location: str


# Season 5 Events (2022-2023)
SEASON_5_EVENTS: List[HyroxEventConfig] = [
    HyroxEventConfig("2EFMS4JI321", 5, "S5 New York 2022", "New York"),
    HyroxEventConfig("2EFMS4JI35D", 5, "S5 Leipzig 2022", "Leipzig"),
    HyroxEventConfig("2EFMS4JI349", 5, "S5 Birmingham 2022", "Birmingham"),
    HyroxEventConfig("2EFMS4JI371", 5, "S5 Valencia 2022", "Valencia"),
    HyroxEventConfig("2EFMS4JI385", 5, "S5 Amsterdam 2022", "Amsterdam"),
    HyroxEventConfig("JGDMS4JI39A", 5, "S5 Chicago 2022", "Chicago"),
    HyroxEventConfig("2EFMS4JI399", 5, "S5 Berlin 2022", "Berlin"),
    HyroxEventConfig("2EFMS4JI3AD", 5, "S5 London 2022", "London"),
    HyroxEventConfig("JGDMS4JI3FE", 5, "S5 Los Angeles 2022", "Los Angeles"),
    HyroxEventConfig("JGDMS4JI3D7", 5, "S5 Dallas 2022", "Dallas"),
    HyroxEventConfig("JGDMS4JI3E9", 5, "S5 Hamburg 2022", "Hamburg"),
    HyroxEventConfig("JGDMS4JI439", 5, "S5 Glasgow 2023", "Glasgow"),
    HyroxEventConfig("JGDMS4JI425", 5, "S5 Manchester 2023", "Manchester"),
    HyroxEventConfig("JGDMS4JI467", 5, "S5 Stockholm 2023", "Stockholm"),
    HyroxEventConfig("JGDMS4JI466", 5, "S5 Barcelona 2023", "Barcelona"),
]

# Season 6 Events (2023)
SEASON_6_EVENTS: List[HyroxEventConfig] = [
    HyroxEventConfig("JGDMS4JI471", 6, "S6 New York 2023", "New York"),
    HyroxEventConfig("JGDMS4JI62E", 6, "S6 London 2023", "London"),
    HyroxEventConfig("JGDMS4JI579", 6, "S6 Paris 2023", "Paris"),
    HyroxEventConfig("JGDMS4JI5E2", 6, "S6 Birmingham 2023", "Birmingham"),
    HyroxEventConfig("JGDMS4JI606", 6, "S6 Amsterdam 2023", "Amsterdam"),
    HyroxEventConfig("JGDMS4JI5C9", 6, "S6 Munich 2023", "Munich"),
    HyroxEventConfig("JGDMS4JI655", 6, "S6 Frankfurt 2023", "Frankfurt"),
    HyroxEventConfig("JGDMS4JI515", 6, "S6 Sydney 2023", "Sydney"),
    HyroxEventConfig("JGDMS4JI516", 6, "S6 Melbourne 2023", "Melbourne"),
    HyroxEventConfig("JGDMS4JI58D", 6, "S6 Singapore 2023", "Singapore"),
]

# All curated events for scraping
ALL_EVENTS: List[HyroxEventConfig] = SEASON_5_EVENTS + SEASON_6_EVENTS


def get_events(
    seasons: List[int] = None,
    locations: List[str] = None,
    limit: int = None
) -> List[HyroxEventConfig]:
    """
    Get filtered list of events for scraping.

    Args:
        seasons: Filter by season numbers (e.g., [5, 6])
        locations: Filter by location names
        limit: Maximum number of events to return

    Returns:
        List of HyroxEventConfig objects matching filters
    """
    events = ALL_EVENTS.copy()

    if seasons:
        events = [e for e in events if e.season in seasons]

    if locations:
        locations_lower = [loc.lower() for loc in locations]
        events = [e for e in events if e.location.lower() in locations_lower]

    if limit:
        events = events[:limit]

    return events


def get_event_by_name(name: str) -> HyroxEventConfig:
    """Get a single event by its name."""
    for event in ALL_EVENTS:
        if event.name == name:
            return event
    return None
