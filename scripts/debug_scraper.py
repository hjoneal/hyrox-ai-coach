"""Debug script to check URL structure for different seasons."""

import sys
sys.path.insert(0, "/Users/harryneal/Documents/personal_projects/hyrox-ai-coach")

import requests
from bs4 import BeautifulSoup

def check_season_url(season: int, event_id: str):
    """Check what URLs are returned for a given season/event."""
    base_url = f"https://results.hyrox.com/season-{season}/"

    session = requests.Session()
    session.get(base_url)

    # Build event URL
    full_event_id = f"H_{event_id}"
    url = f"{base_url}?page=1&event={full_event_id}&num_results=100&pid=list&pidp=start&ranking=time_finish_netto"

    print(f"\n{'='*60}")
    print(f"Season {season} - Event: {event_id}")
    print(f"URL: {url}")

    response = session.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find first participant link
    rows = soup.find_all("li", class_="row")
    for row in rows[:3]:
        link_tag = row.find("a", href=lambda x: x and "content=detail" in x)
        if link_tag:
            href = link_tag.get('href', 'No href')
            print(f"  Sample href: {href[:100]}...")
            break

    # Check page title/header
    h2 = soup.find('h2')
    if h2:
        print(f"  Page title: {h2.text.strip()[:80]}...")

# Test Season 8 (known working)
check_season_url(8, "LR3MS4JI11FA")

# Test Season 5
check_season_url(5, "2EFMS4JI321")

# Test Season 6
check_season_url(6, "JGDMS4JI471")
