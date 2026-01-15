import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import re
import time
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyroxScraper:
    def __init__(self, season="season-8"):
        self.base_url = f"https://results.hyrox.com/{season}/"
        self.session = requests.Session()
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
                    profile_link = f"https://results.hyrox.com{href}" if href.startswith("/") else href
                    if not profile_link.startswith("http"):
                        profile_link = f"https://results.hyrox.com/{self.session.cookies.get('season', 'season-8')}/{href}"

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

if __name__ == "__main__":
    # Test with the Event ID you found
    scraper = HyroxScraper(season="season-8")
    target_event = "H_LR3MS4JI11FA" # Stockholm Open
    
    print(f"🚀 Starting test scrape for: {target_event}")
    df = scraper.scrape_event(target_event, limit_pages=1)
    
    if not df.empty:
        print(f"✅ Success! Scraped {len(df)} rows.")
        # Print columns to verify we have run_1, station_1, etc.
        print(df.columns)
        print(df.head(2))
        
        output_file = "data/raw/test_scrape_v2.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved to {output_file}")
    else:
        print("❌ Still no data. We might need to check the link construction.")