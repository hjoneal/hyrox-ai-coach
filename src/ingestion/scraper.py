import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import re
import time
import os
from typing import List, Optional
from tqdm import tqdm

from src.ingestion.events import HyroxEventConfig, Division

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyroxScraper:
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
        try:
            self.session.get(self.base_url)
            logging.info(f"Session initialized with {self.base_url}")
        except Exception as e:
            logging.error(f"Failed to initialize session: {e}")

    def get_html(self, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    def parse_time_str(self, time_str):
        """Converts strings like '00:04:30' or '1:05:00' to Timedelta."""
        if not time_str or "–" in time_str:
            return timedelta()
        try:
            parts = list(map(int, time_str.split(":")))
            if len(parts) == 2: # mm:ss
                return timedelta(minutes=parts[0], seconds=parts[1])
            elif len(parts) == 3: # hh:mm:ss
                return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
        except ValueError:
            pass
        return timedelta()

    def parse_splits(self, participant_link):
        """
        Extracts splits from the Detail Page using the Season 8 CSS classes.
        Mapping: f-time_0X = Run X, f-time_1X = Station X.
        """
        html = self.get_html(participant_link)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')
        splits = {}

        # The splits are in a table usually found in the right-hand channel or 'box-other'
        # We iterate over all table rows to find the specific classes.
        all_rows = soup.find_all("tr")

        for row in all_rows:
            # Get the class list of the row (e.g., ['f-time_01', 'list-highlight'])
            classes = row.get("class", [])
            if not classes:
                continue

            # Check if this row is a time row
            row_class = next((c for c in classes if c.startswith("f-time_")), None)
            
            if row_class:
                # The value is usually in the second cell (td) with the SAME class
                # e.g., <td class="f-time_01">00:03:06</td>
                cells = row.find_all("td")
                if not cells: continue
                
                # Find the cell that has the matching class or is the second cell
                val_cell = row.find("td", class_=row_class)
                if not val_cell and len(cells) > 1:
                    val_cell = cells[0] # Fallback
                
                if val_cell:
                    time_val = self.parse_time_str(val_cell.text.strip())
                    
                    # --- MAPPING LOGIC ---
                    # f-time_01 -> Run 1
                    # f-time_08 -> Run 8
                    # f-time_11 -> Station 1 (Ski)
                    # f-time_18 -> Station 8 (Wallballs)
                    
                    suffix = row_class.replace("f-time_", "")
                    
                    if suffix.startswith("0"): # Runs
                        run_num = int(suffix[1])
                        splits[f"run_{run_num}"] = time_val
                    
                    elif suffix.startswith("1"): # Stations
                        station_num = int(suffix[1])
                        splits[f"station_{station_num}"] = time_val
                        
                    elif suffix == "finish_netto":
                        splits["total_time"] = time_val

        # Ensure we have data
        if not splits:
            logging.warning(f"No splits found for {participant_link}")
            return None

        # Calculate Totals for convenience
        # Note: We use .get() with default 0 timedelta to avoid errors if a split is missing
        runs = [splits.get(f"run_{i}", timedelta()) for i in range(1, 9)]
        stations = [splits.get(f"station_{i}", timedelta()) for i in range(1, 9)]
        
        splits["total_run"] = sum(runs, timedelta())
        splits["total_stations"] = sum(stations, timedelta())
        
        # If total_time wasn't found in table, sum the parts + roxzone (roxzone is tricky to calculate without direct data, effectively it's Total - (Runs + Stations))
        if "total_time" not in splits:
             # Try to find it in the main header if missing
             pass 

        return splits

    def scrape_event(self, event_id, limit_pages=None):
        page = 1
        all_participants = []
        
        logging.info(f"Starting scrape for Event ID: {event_id}")

        while True:
            url = f"{self.base_url}?page={page}&event={event_id}&num_results=100&pid=list&pidp=start&ranking=time_finish_netto"
            html = self.get_html(url)
            if not html: break
            
            soup = BeautifulSoup(html, 'html.parser')
            rows = soup.find_all("li", class_="row")
            
            # Filter rows that look like data (have a data-id or detail link)
            data_rows = []
            for r in rows:
                if r.find("a", href=lambda x: x and "content=detail" in x):
                    data_rows.append(r)

            if not data_rows:
                logging.info("No more participant rows found.")
                break

            logging.info(f"Processing Page {page} with {len(data_rows)} participants...")

            for row in data_rows:
                try:
                    # 1. Get Link
                    link_tag = row.find("a", href=lambda x: x and "content=detail" in x)
                    if not link_tag: continue
                    
                    href = link_tag['href']
                    # Construct profile link using current base_url
                    if href.startswith("http"):
                        profile_link = href
                    elif href.startswith("/"):
                        profile_link = f"https://results.hyrox.com{href}"
                    else:
                        # Relative URL - append to base_url
                        profile_link = f"{self.base_url}{href}"

                    # 2. Get Name (Try a few common classes)
                    name = "Unknown"
                    name_div = row.find(class_="type-fullname") or row.find(class_="f-__fullname_last_first")
                    if name_div:
                        name = name_div.get_text(strip=True)

                    # 3. Deep Dive
                    splits = self.parse_splits(profile_link)
                    
                    if splits:
                        record = {
                            "event_id": event_id,
                            "name": name,
                            "link": profile_link,
                            **splits
                        }
                        all_participants.append(record)
                
                except Exception as e:
                    logging.error(f"Error parsing row: {e}")
                    continue

            page += 1
            if limit_pages and page > limit_pages:
                break
            time.sleep(1)

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

        for event in tqdm(events, desc="Scraping events"):
            # Check if we already have this event scraped
            safe_name = event.name.replace(" ", "_").replace("/", "-")
            event_file = os.path.join(output_dir, f"{safe_name}.csv")

            if save_intermediate and os.path.exists(event_file):
                logging.info(f"Loading cached data for {event.name}")
                df = pd.read_csv(event_file)
                all_data.append(df)
                continue

            # Switch season if needed
            if self.season != event.season:
                self.set_season(event.season)

            # Build full event ID with division prefix
            full_event_id = f"{division.value}_{event.event_id}"

            try:
                logging.info(f"Scraping {event.name} (ID: {full_event_id})")
                df = self.scrape_event(full_event_id)

                if df.empty:
                    logging.warning(f"No data returned for {event.name}")
                    failed_events.append(event.name)
                    continue

                # Add event metadata
                df["event_name"] = event.name
                df["season"] = event.season
                df["location"] = event.location

                # Validate data
                df = self._validate_event_data(df)

                all_data.append(df)

                # Save intermediate results
                if save_intermediate:
                    df.to_csv(event_file, index=False)
                    logging.info(f"Saved {len(df)} rows to {event_file}")

            except Exception as e:
                logging.error(f"Failed to scrape {event.name}: {e}")
                failed_events.append(event.name)
                continue

        if failed_events:
            logging.warning(f"Failed events: {failed_events}")

        if not all_data:
            logging.error("No data scraped from any event")
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        logging.info(f"Total participants scraped: {len(combined)}")

        return combined

    def _validate_event_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean scraped event data.

        Args:
            df: Raw scraped DataFrame

        Returns:
            Cleaned DataFrame with invalid rows removed
        """
        original_len = len(df)

        # Check for required columns
        required_cols = ["run_1", "station_1", "total_time"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}")
            return df

        # Remove rows with missing critical data
        df = df.dropna(subset=["total_time"])

        # Remove rows where total_time is zero or negative
        # (timedelta strings that start with "0 days 00:00:00")
        df = df[df["total_time"] != "0 days 00:00:00"]

        removed = original_len - len(df)
        if removed > 0:
            logging.info(f"Removed {removed} invalid rows")

        return df


if __name__ == "__main__":
    # Test single event scraping
    scraper = HyroxScraper(season=8)
    target_event = "H_LR3MS4JI11FA"  # Stockholm Open

    print(f"Starting test scrape for: {target_event}")
    df = scraper.scrape_event(target_event, limit_pages=1)

    if not df.empty:
        print(f"Success! Scraped {len(df)} rows.")
        print(df.columns.tolist())
        print(df.head(2))

        output_file = "data/raw/test_scrape_v2.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
    else:
        print("No data returned. Check the event ID and connection.")