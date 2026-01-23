"""
Hyrox Results Scraper

Scrapes participant data and splits from the official Hyrox results website.
All times are stored as integers (seconds) for clean data format.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import logging
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from src.ingestion.events import HyroxEventConfig, Division

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_time_to_seconds(time_str: str) -> int:
    """
    Parse time string to seconds.

    Args:
        time_str: Time in format 'HH:MM:SS', 'MM:SS', or 'H:MM:SS'

    Returns:
        Integer seconds, or 0 if invalid
    """
    if not time_str or time_str.strip() in ('', '-', '–', '--'):
        return 0

    time_str = time_str.strip()

    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except (ValueError, AttributeError):
        pass

    return 0


class HyroxScraper:
    """Scraper for Hyrox competition results."""

    def __init__(self, season: int = 8):
        """
        Initialize the Hyrox scraper.

        Args:
            season: Season number (e.g., 5, 6, 8)
        """
        self.season = season
        self.base_url = f"https://results.hyrox.com/season-{season}/"
        self.session = requests.Session()
        self._init_session()

    def set_season(self, season: int):
        """Switch to a different season."""
        self.season = season
        self.base_url = f"https://results.hyrox.com/season-{season}/"
        self._init_session()

    def _init_session(self):
        """Initialize HTTP session."""
        try:
            self.session.get(self.base_url, timeout=10)
            logger.info(f"Session initialized: season-{self.season}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")

    def _get_html(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"HTTP error: {e}")
            return None

    def _get_td_text(self, container, class_name: str) -> str:
        """Extract text from td element with given class."""
        if not container:
            return ''
        td = container.find('td', class_=class_name)
        if td:
            # Get text, handling nested spans
            text = td.get_text(strip=True)
            return text if text and text not in ('-', '–', '--') else ''
        return ''

    def _get_td_int(self, container, class_name: str) -> int:
        """Extract integer from td element with given class."""
        text = self._get_td_text(container, class_name)
        if not text:
            return 0
        try:
            # Remove common formatting (commas, dots for thousands)
            cleaned = text.replace(',', '').replace('.', '').strip()
            return int(cleaned)
        except ValueError:
            return 0

    def _get_time_seconds(self, container, class_name: str) -> int:
        """Extract time as seconds from td element with given class."""
        text = self._get_td_text(container, class_name)
        return parse_time_to_seconds(text)

    def parse_participant_details(self, profile_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract all participant data from their detail page.

        Args:
            profile_url: URL to participant's detail page

        Returns:
            Dictionary with all participant data, or None if failed
        """
        html = self._get_html(profile_url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')
        data = {}

        # =====================================================================
        # LEFT COLUMN: Participant details from untabbed-boxes
        # =====================================================================

        # Participant box (#detail-box-general)
        participant_box = soup.find('div', id='detail-box-general')
        if participant_box:
            data['bib_number'] = self._get_td_text(participant_box, 'f-start_no_text')
            data['age_group'] = self._get_td_text(participant_box, 'f-type_age_class')
            data['nationality'] = self._get_td_text(participant_box, 'f--nation')

        # Race details box (#detail-box-eventinfo)
        event_box = soup.find('div', id='detail-box-eventinfo')
        if event_box:
            data['division'] = self._get_td_text(event_box, 'f--event')

        # Judging box (#detail-box-judges)
        judges_box = soup.find('div', id='detail-box-judges')
        if judges_box:
            data['bonus_time'] = self._get_time_seconds(judges_box, 'f-gimmick_04')
            data['penalty_time'] = self._get_time_seconds(judges_box, 'f-gimmick_01')
            data['disqual_reason'] = self._get_td_text(judges_box, 'f-disqual_reason')

        # Overall results box (#detail-box-totals)
        totals_box = soup.find('div', id='detail-box-totals')
        if totals_box:
            data['rank_overall'] = self._get_td_int(totals_box, 'f-place_all')
            data['rank_age_group'] = self._get_td_int(totals_box, 'f-place_age')
            data['overall_time'] = self._get_time_seconds(totals_box, 'f-time_finish_netto')

        # =====================================================================
        # RIGHT COLUMN: Splits from workout summary
        # =====================================================================

        all_rows = soup.find_all('tr')

        for row in all_rows:
            classes = row.get('class', [])
            if not classes:
                continue

            # Find time class (e.g., f-time_01, f-time_11, f-time_60)
            time_class = next((c for c in classes if c.startswith('f-time_')), None)
            if not time_class:
                continue

            # Get the time value from td with matching class
            td = row.find('td', class_=time_class)
            if not td:
                continue

            time_seconds = parse_time_to_seconds(td.get_text(strip=True))
            suffix = time_class.replace('f-time_', '')

            # Map suffix to field name
            # f-time_01 to f-time_08 = Run 1-8
            # f-time_11 to f-time_18 = Station 1-8
            # f-time_49 = Run Total
            # f-time_50 = Best Run Lap
            # f-time_60 = Roxzone Time

            if suffix.isdigit():
                num = int(suffix)
                if 1 <= num <= 8:  # Runs
                    data[f'run_{num}'] = time_seconds
                elif 11 <= num <= 18:  # Stations
                    data[f'station_{num - 10}'] = time_seconds
                elif num == 49:
                    data['total_run'] = time_seconds
                elif num == 50:
                    data['best_run_lap'] = time_seconds
                elif num == 60:
                    data['roxzone_time'] = time_seconds

        # Calculate totals if not found
        if 'total_run' not in data:
            data['total_run'] = sum(data.get(f'run_{i}', 0) for i in range(1, 9))

        if 'total_stations' not in data:
            data['total_stations'] = sum(data.get(f'station_{i}', 0) for i in range(1, 9))

        # Validate we have actual split data
        has_runs = any(data.get(f'run_{i}', 0) > 0 for i in range(1, 9))
        has_stations = any(data.get(f'station_{i}', 0) > 0 for i in range(1, 9))

        if not has_runs and not has_stations:
            return None

        return data

    def scrape_event(
        self,
        event_id: str,
        limit_pages: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Scrape all participants from a single event.

        Args:
            event_id: Event ID (e.g., 'H_LR3MS4JI11FA')
            limit_pages: Maximum pages to scrape (None for all)

        Returns:
            DataFrame with all participant data
        """
        page = 1
        all_participants = []
        total_scraped = 0

        logger.info(f"Starting scrape: {event_id}")

        while True:
            url = (
                f"{self.base_url}?page={page}&event={event_id}"
                f"&num_results=100&pid=list&pidp=start&ranking=time_finish_netto"
            )

            html = self._get_html(url)
            if not html:
                break

            soup = BeautifulSoup(html, 'html.parser')

            # Find participant rows (li elements with detail links)
            rows = soup.find_all('li', class_='row')
            data_rows = [
                r for r in rows
                if r.find('a', href=lambda x: x and 'content=detail' in x)
            ]

            if not data_rows:
                logger.info(f"Page {page}: No more participants")
                break

            page_count = len(data_rows)
            logger.info(f"Page {page}: Processing {page_count} participants...")

            for i, row in enumerate(data_rows, 1):
                try:
                    # Get participant link
                    link_tag = row.find('a', href=lambda x: x and 'content=detail' in x)
                    if not link_tag:
                        continue

                    href = link_tag['href']
                    if href.startswith('http'):
                        profile_link = href
                    elif href.startswith('/'):
                        profile_link = f"https://results.hyrox.com{href}"
                    else:
                        profile_link = f"{self.base_url}{href}"

                    # Get name from list view
                    name = "Unknown"
                    name_div = (
                        row.find(class_='type-fullname') or
                        row.find(class_='f-__fullname_last_first')
                    )
                    if name_div:
                        name = name_div.get_text(strip=True)

                    # Scrape detail page
                    details = self.parse_participant_details(profile_link)

                    if details:
                        record = {
                            'event_id': event_id,
                            'name': name,
                            'link': profile_link,
                            **details
                        }
                        all_participants.append(record)
                        total_scraped += 1

                        # Progress logging every 25 participants
                        if total_scraped % 25 == 0:
                            logger.info(f"  ... scraped {total_scraped} participants")

                except Exception as e:
                    logger.warning(f"Error parsing participant: {e}")
                    continue

            page += 1
            if limit_pages and page > limit_pages:
                break

            time.sleep(1)  # Rate limiting

        logger.info(f"Completed: {total_scraped} participants scraped")
        return pd.DataFrame(all_participants)

    def scrape_multiple_events(
        self,
        events: List[HyroxEventConfig],
        division: Division = Division.OPEN,
        output_dir: str = "data/raw/events",
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Scrape multiple events with progress tracking and resumability.

        Args:
            events: List of HyroxEventConfig objects to scrape
            division: Division to scrape (default: Open)
            output_dir: Directory for intermediate saves
            save_intermediate: Whether to save each event separately

        Returns:
            Combined DataFrame with all participants from all events
        """
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)

        all_data = []
        failed_events = []
        total_participants = 0

        logger.info(f"=" * 60)
        logger.info(f"BATCH SCRAPE: {len(events)} events")
        logger.info(f"=" * 60)

        for idx, event in enumerate(events, 1):
            # Check for cached data (resumability)
            safe_name = event.name.replace(" ", "_").replace("/", "-")
            event_file = os.path.join(output_dir, f"{safe_name}.csv")

            logger.info(f"")
            logger.info(f"[{idx}/{len(events)}] {event.name}")

            if save_intermediate and os.path.exists(event_file):
                logger.info(f"  -> Loading from cache: {event_file}")
                df = pd.read_csv(event_file)
                all_data.append(df)
                total_participants += len(df)
                logger.info(f"  -> Loaded {len(df)} participants (total: {total_participants})")
                continue

            # Switch season if needed
            if self.season != event.season:
                self.set_season(event.season)

            # Build full event ID with division prefix
            full_event_id = f"{division.value}_{event.event_id}"

            try:
                df = self.scrape_event(full_event_id)

                if df.empty:
                    logger.warning(f"  -> No data returned")
                    failed_events.append(event.name)
                    continue

                # Add event metadata
                df['event_name'] = event.name
                df['season'] = event.season
                df['location'] = event.location

                # Validate and clean
                df = self._validate_event_data(df)

                all_data.append(df)
                total_participants += len(df)

                # Save intermediate results
                if save_intermediate:
                    df.to_csv(event_file, index=False)
                    logger.info(f"  -> Saved {len(df)} participants to {event_file}")

                logger.info(f"  -> Total so far: {total_participants}")

            except Exception as e:
                logger.error(f"  -> Failed: {e}")
                failed_events.append(event.name)
                continue

        logger.info(f"")
        logger.info(f"=" * 60)
        logger.info(f"SCRAPE COMPLETE")
        logger.info(f"=" * 60)
        logger.info(f"Total participants: {total_participants}")
        logger.info(f"Successful events: {len(events) - len(failed_events)}/{len(events)}")

        if failed_events:
            logger.warning(f"Failed events: {failed_events}")

        if not all_data:
            logger.error("No data scraped from any event")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        return combined

    def _validate_event_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean scraped event data.

        Removes rows with:
        - Missing critical split data
        - Zero total time
        - Unreasonable times (< 30 min or > 4 hours)
        """
        original_len = len(df)

        # Remove rows with zero overall time
        if 'overall_time' in df.columns:
            df = df[df['overall_time'] > 0]

        # Remove unreasonable times (< 30 min or > 4 hours)
        if 'overall_time' in df.columns:
            min_time = 30 * 60   # 30 minutes
            max_time = 4 * 3600  # 4 hours
            df = df[(df['overall_time'] >= min_time) & (df['overall_time'] <= max_time)]

        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"  -> Removed {removed} invalid rows")

        return df


if __name__ == "__main__":
    # Test single participant scrape
    scraper = HyroxScraper(season=8)

    # Test with Stockholm Open
    test_url = (
        "https://results.hyrox.com/season-8/"
        "?content=detail&fpid=list&pid=list&idp=LR3MS4JI44A6FC"
        "&lang=EN_CAP&event=H_LR3MS4JI11FA&num_results=100"
        "&pidp=start&ranking=time_finish_netto&search_event=H_LR3MS4JI11FA"
    )

    print("Testing single participant scrape...")
    details = scraper.parse_participant_details(test_url)

    if details:
        print("\nExtracted fields:")
        for key, value in sorted(details.items()):
            print(f"  {key}: {value}")
    else:
        print("Failed to extract data")
