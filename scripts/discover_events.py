"""Discover which events have Mens Pro Doubles (HDP) data.

Probes each known event ID with the HDP division prefix to check
if it returns valid Pro Doubles results. Reports which events are
available for scraping.
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup
import time as time_mod

from src.ingestion.events import (
    ALL_EVENTS, SEASON_5_EVENTS, SEASON_6_EVENTS,
    SEASON_7_EVENTS, SEASON_8_EVENTS,
    Division, Gender,
)

DIVISION = Division.PRO_DOUBLES
GENDER = Gender.MEN


def probe_event(event, session):
    """Check if an event has Pro Doubles data. Returns participant count or 0."""
    base_url = f"https://results.hyrox.com/season-{event.season}/"
    full_event_id = f"{DIVISION.value}_{event.event_id}"

    url = (
        f"{base_url}?page=1&event={full_event_id}"
        f"&num_results=100&pid=list&pidp=start&ranking=time_finish_netto"
    )
    if GENDER.value:
        url += f"&sex={GENDER.value}"

    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return -1, str(e)

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Count participant rows with detail links
    rows = soup.find_all('li', class_='row')
    data_rows = [
        r for r in rows
        if r.find('a', href=lambda x: x and 'content=detail' in x)
    ]

    return len(data_rows), None


def main():
    season_groups = [
        ("Season 5", SEASON_5_EVENTS),
        ("Season 6", SEASON_6_EVENTS),
        ("Season 7", SEASON_7_EVENTS),
        ("Season 8", SEASON_8_EVENTS),
    ]

    session = requests.Session()
    # Warm up session
    session.get("https://results.hyrox.com/season-5/", timeout=10)

    valid_events = []

    for group_name, events in season_groups:
        print(f"\n{'='*60}")
        print(f"Probing {group_name} ({len(events)} events) for {DIVISION.name} {GENDER.name}")
        print(f"{'='*60}")

        for event in events:
            count, error = probe_event(event, session)

            if error:
                status = f"ERROR: {error}"
            elif count > 0:
                status = f"FOUND {count} participants"
                valid_events.append(event)
            else:
                status = "No data"

            print(f"  {event.name:30s} -> {status}")
            time_mod.sleep(0.5)  # Rate limiting

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(valid_events)}/{len(ALL_EVENTS)} events have {DIVISION.name} {GENDER.name} data")
    print(f"{'='*60}")

    if valid_events:
        print("\nValid events for scraping:")
        for e in valid_events:
            print(f"  - {e.name} (Season {e.season}, ID: {e.event_id})")

        print("\nPython list for copy/paste:")
        print("DOUBLES_EVENTS = [")
        for e in valid_events:
            print(f'    HyroxEventConfig("{e.event_id}", {e.season}, "{e.name}", "{e.location}"),')
        print("]")


if __name__ == "__main__":
    main()
