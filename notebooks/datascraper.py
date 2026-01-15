# This notebook enables you to scrape the results from the Hyrox results website (https://hyrox.r.mikatiming.com/season-5/?pid=start&pidp=start)

# Each event needs to be scraped individually and to do this, the event id must be found. To do this, go to the results website linked above, select the desired event and show the any result for that event. This will take you to list view with results. Next scroll to the bottom of the page and press the reload button. This will update the page url to include the event id for the selected event

# For example, carrying out the necessary steps for the Season 5 2023 Glasgow event results in the following url being shown - https://hyrox.r.mikatiming.com/season-5/?event=HPRO_JGDMS4JI439. The event id for this event would be 'JGDMS4JI439'. This id can then be inputted into the HyroxEvent class and all results will be scraped into a csv file.

# All events with their event ids that are already part of the dataset can be found further down in this notebook.

import os
from requests import request
from bs4 import BeautifulSoup
from enum import Enum
import re
import itertools
from tqdm import tqdm
from datetime import timedelta
import numpy as np
import pandas as pd

class Division(Enum):
    open = "H"
    pro = "HPRO"
    elite = "HE"
    doubles = "HD"
    relay = "HMR"
    goruck = "HG"
    goruck_doubles = "HDG"

    class Gender(Enum):
    male = "M"
    female = "W"

    def get_html(url: str):
    cookie_retrieval = request("GET", url)
    cookie = cookie_retrieval.request.headers.get("Cookie")
    response = request("GET", url, headers={"Cookie": cookie})
    return response.text

def removeprefix(x: str, prefix: str):
    if x.startswith(prefix):
        return x[len(prefix):]
    return x

class HyroxParticipant:
    def __init__(self, id: str, name: str, division: Division, gender: Gender, age_group: str, time: str, link: str):
        self.id = id
        self.name = name
        self.division = division
        self.gender = gender
        self.age_group = age_group
        self.time = time
        self.link = link
        self._raw_splits = []
        self.splits = {}
        self.ignore = False

    def get_timings(self):
        html = get_html(self.link)
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            splits = soup.find(class_="detail-box box-splits")
            split_rows = splits.find("tbody").find_all("tr")
        except AttributeError:
            self.ignore = True
            return

        def get_diff(i: int):
            if i >= len(split_rows):
                return timedelta()
            
            raw = split_rows[i].select("td.diff")[0].text
            if "–" in raw:
                return timedelta()
            else:
                return timedelta(minutes=int(raw.split(":")[0]), seconds=int(raw.split(":")[1]))

        def get_details():
            skierg_rox_in = self._raw_splits[0]
            skierg_in = self._raw_splits[1]
            skierg_out = self._raw_splits[2]
            skierg_rox_out = self._raw_splits[3]

            push_rox_in = self._raw_splits[4]
            push_in = self._raw_splits[5]
            push_out = self._raw_splits[6]
            push_rox_out = self._raw_splits[7]

            pull_rox_in = self._raw_splits[8]
            pull_in = self._raw_splits[9]
            pull_out = self._raw_splits[10]
            pull_rox_out = self._raw_splits[11]

            burpee_rox_in = self._raw_splits[12]
            burpee_in = self._raw_splits[13]
            burpee_out = self._raw_splits[14]
            burpee_rox_out = self._raw_splits[15]

            row_rox_in = self._raw_splits[16]
            row_in = self._raw_splits[17]
            row_out = self._raw_splits[18]
            row_rox_out = self._raw_splits[19]

            carry_rox_in = self._raw_splits[20]
            carry_in = self._raw_splits[21]
            carry_out = self._raw_splits[22]
            carry_rox_out = self._raw_splits[23]

            lunges_rox_in = self._raw_splits[24]
            lunges_in = self._raw_splits[25]
            lunges_out = self._raw_splits[26]
            lunges_rox_out = self._raw_splits[27]

            wall_in = self._raw_splits[28]
            finish = self._raw_splits[29]

            self.splits["total"] = np.sum(self._raw_splits)
            self.splits["stations"] = [skierg_out, push_out, pull_out, burpee_out, row_out, carry_out, lunges_out, finish]
            self.splits["work"] = np.sum(self.splits["stations"])
            self.splits["runs"] = [skierg_rox_in, push_rox_in, pull_rox_in, burpee_rox_in, row_rox_in, carry_rox_in, lunges_rox_in, wall_in]
            self.splits["run"] = np.sum(self.splits["runs"])
            self.splits["rests"] = [skierg_in + skierg_rox_out, push_in + push_rox_out, pull_in + pull_rox_out, burpee_in + burpee_rox_out, row_in + row_rox_out, carry_in + carry_rox_out, lunges_in + lunges_rox_out, timedelta()]
            self.splits["roxzone"] = np.sum(self.splits["rests"])


        self._raw_splits = [get_diff(x) for x in range(30)]
        get_details()

    def to_array(self):
        return [
            self.id,
            self.name,
            self.gender.name,
            self.age_group,
            self.division.name,
            str(self.splits["total"]),
            str(self.splits["work"]),
            str(self.splits["roxzone"]),
            str(self.splits["run"]),
            str(self.splits["runs"][0]),
            str(self.splits["stations"][0]),
            str(self.splits["rests"][0]),
            str(self.splits["runs"][1]),
            str(self.splits["stations"][1]),
            str(self.splits["rests"][1]),
            str(self.splits["runs"][2]),
            str(self.splits["stations"][2]),
            str(self.splits["rests"][2]),
            str(self.splits["runs"][3]),
            str(self.splits["stations"][3]),
            str(self.splits["rests"][3]),
            str(self.splits["runs"][4]),
            str(self.splits["stations"][4]),
            str(self.splits["rests"][4]),
            str(self.splits["runs"][5]),
            str(self.splits["stations"][5]),
            str(self.splits["rests"][5]),
            str(self.splits["runs"][6]),
            str(self.splits["stations"][6]),
            str(self.splits["rests"][6]),
            str(self.splits["runs"][7]),
            str(self.splits["stations"][7]),
            str(self.splits["rests"][7]),
        ]
    
    class HyroxEvent:
    def __init__(self, event_id: str, season: int):
        self.event_id = event_id
        self.season = season
        self.event_name = ""
        self.event_participants = dict((x, dict((y, []) for y in Gender)) for x in Division)
        self.num_event_participants = dict((x, dict((y, 0) for y in Gender)) for x in Division)

    @property
    def participants(self):
        arr = []
        for division, arr2 in self.event_participants.items():
            for gender, arr3 in arr2.items():
                arr.extend(arr3)
        return arr

    def generate_url(self, page: int, division: Division, gender: Gender):
        return f"https://hyrox.r.mikatiming.com/season-{self.season}/?page={page}&event={division.value}_{self.event_id}&num_results=100&pid=list&pidp=start&ranking=time_finish_netto&search%5Bsex%5D={gender.value}&search%5Bage_class%5D=%25&search%5Bnation%5D=%25"

    def get_info(self):
        combinations = list(itertools.product(Division, Gender))
        pbar = tqdm(combinations, desc="Retrieving participants")
        for division, gender in pbar:
            page = 1
            while True:
                pbar.set_postfix({
                    "Division": division.name,
                    "Gender": gender.name,
                    "Page": page
                })
                url = self.generate_url(page, division=division, gender=gender)
                html = get_html(url)
                soup = BeautifulSoup(html, 'html.parser')

                h2_title: str = soup.h2.text.strip()
                h2_title = removeprefix(h2_title, "Results: ").split(" / HYROX")[0]
                if h2_title == "General Ranking / All":
                    break
                    
                list_headers = soup.select(".list-group-header .list-field")
                if len(list_headers) > 0 and list_headers[0].text.strip() == "Race":
                    break
                    
                if self.event_name == "":
                    self.event_name = f"S{self.season} {h2_title.strip()}"
                    if self.event_name == "S4 WorldChampionship - Leipzig":
                        self.event_name = "S4 2021 Leipzig - World Championship"

                if self.num_event_participants[division][gender] == 0:
                    list_info: str = soup.find(class_="list-info").li.text
                    num_participants = int(re.findall("(\d+) Results", list_info)[0])
                    self.num_event_participants[division][gender] = num_participants

                list_rows = list(soup.find_all("li", class_="list-group-item row"))
                list_rows.extend(list(soup.find_all("li", class_="list-active list-group-item row")))
            
                for row in list_rows:
                    fields = row.select(".list-field")
                    
                    id = ""
                    name = ""
                    age_group = ""
                    time = ""
                    link = ""
                    
                    for field_i in range(len(list_headers)):
                        header = "".join(list_headers[field_i].find_all(string=True, recursive=False))
                        field = fields[field_i]
                        field_content = field.text.strip()
                        if header == "Number":
                            id = removeprefix(field_content, "Number")
                        elif header == "Age Group":
                            age_group = removeprefix(field_content, "Age Group")
                            if age_group == "–":
                                age_group = ""
                        elif header == "Name" or header == "Member":
                            name = field_content
                            link = f"https://hyrox.r.mikatiming.com/season-{self.season}/{field.a.get('href')}"
                        elif header == "Total":
                            time = removeprefix(field_content, "Total")
                    
                    participant = HyroxParticipant(
                        id=id,
                        name=name,
                        division=division,
                        gender=gender,
                        age_group=age_group,
                        time=time,
                        link=link
                    )
                    self.event_participants[division][gender].append(participant)

                if len(self.event_participants[division][gender]) < self.num_event_participants[division][gender]:
                    page += 1
                else:
                    break

        for participant in tqdm(self.participants, desc="Retrieving splits"):
            participant.get_timings()

        def participant_filter(p: HyroxParticipant):
            if p.ignore:
                return False
            
            if any(list(map(lambda x: x < timedelta(), p.splits["stations"]))):
                return False

            if any(list(map(lambda x: x < timedelta(), p.splits["runs"]))):
                return False

            if any(list(map(lambda x: x < timedelta(), p.splits["rests"]))):
                return False

            return True

        for division, gender in combinations:
            self.event_participants[division][gender] = filter(participant_filter, self.event_participants[division][gender])

    def save(self, directory: str = "/kaggle/working"):

        def participant_map(p: HyroxParticipant):
            arr = [self.event_id, self.event_name]
            arr.extend(p.to_array())
            return np.array(arr)

        df = pd.DataFrame(
            np.array(list(map(participant_map, self.participants))),
            columns=["event_id", "event_name", "id", "name", "gender", "age_group", "division", "total_time", "work_time", "roxzone_time", "run_time", "run_1", "work_1", "roxzone_1", "run_2", "work_2", "roxzone_2", "run_3", "work_3", "roxzone_3", "run_4", "work_4", "roxzone_4", "run_5", "work_5", "roxzone_5", "run_6", "work_6", "roxzone_6", "run_7", "work_7", "roxzone_7", "run_8", "work_8", "roxzone_8"]
        )
                
        df.insert(5, "nationality", df["name"].str.extract(r'\(([A-Z]{3})\)', expand=False))
        df.drop(["id", "name"], axis=1, inplace=True)
        
        df.to_csv(os.path.join(directory, f"{self.event_name}.csv"), index=False)
        
    def copy(self):
        return self.__copy__()
    
    def __copy__(self):
        new_event = HyroxEvent(
            event_id=self.event_id, 
            season=self.season
        )
        new_event.event_name = self.event_name
        new_event.event_participants = self.event_participants
        new_event.num_event_participants = self.num_event_participants
        return new_event
    
    class Event(Enum):
    s4_leipzig2021 = HyroxEvent(event_id="2EFMS4JI259", season=4)
    s4_leipzigWC = HyroxEvent(event_id="2EFMS4JI25B", season=4)
    s4_london2021 = HyroxEvent(event_id="2EFMS4JI26E", season=4)
    s4_losangeles2021 = HyroxEvent(event_id="2EFMS4JI26D", season=4)
    s4_amsterdam2021 = HyroxEvent(event_id="2EFMS4JI281", season=4)
    s4_birmingham2021 = HyroxEvent(event_id="2EFMS4JI282", season=4)
    s4_stuttgart2021 = HyroxEvent(event_id="2EFMS4JI283", season=4)
    s4_berlin2021 = HyroxEvent(event_id="2EFMS4JI285", season=4)
    s4_dallas2021 = HyroxEvent(event_id="2EFMS4JI284", season=4)
    s4_hamburg2021 = HyroxEvent(event_id="2EFMS4JI286", season=4)
    s4_madrid2021 = HyroxEvent(event_id="2EFMS4JI289", season=4)
    s4_manchester2021 = HyroxEvent(event_id="2EFMS4JI2BD", season=4)
    s4_munchen2021 = HyroxEvent(event_id="2EFMS4JI2BE", season=4)
    s4_newyork2021 = HyroxEvent(event_id="2EFMS4JI2C2", season=4)
    s4_bremen2022 = HyroxEvent(event_id="2EFMS4JI2C3", season=4)
    s4_maastricht2022 = HyroxEvent(event_id="2EFMS4JI2C4", season=4)
    s4_karlsruhe2022 = HyroxEvent(event_id="2EFMS4JI2C5", season=4)
    s4_wien2022 = HyroxEvent(event_id="2EFMS4JI2C7", season=4)
    s4_lasvegas2022 = HyroxEvent(event_id="2EFMS4JI2CC", season=4)
    s4_chicago2022 = HyroxEvent(event_id="2EFMS4JI2A9", season=4)
    s4_essen2022 = HyroxEvent(event_id="2EFMS4JI2CA", season=4)
    s4_london2022 = HyroxEvent(event_id="2EFMS4JI2C9", season=4)
    s4_dallas2022 = HyroxEvent(event_id="2EFMS4JI2C6", season=4)
    s4_frankfurt2022 = HyroxEvent(event_id="2EFMS4JI2C8", season=4)
    s4_losangeles2022 = HyroxEvent(event_id="2EFMS4JI2CB", season=4)
    
    s5_basel2022 = HyroxEvent(event_id="2EFMS4JI2BF", season=5)
    s5_newyork2022 = HyroxEvent(event_id="2EFMS4JI321", season=5)
    s5_leipzig2022 = HyroxEvent(event_id="2EFMS4JI35D", season=5)
    s5_birmingham2022 = HyroxEvent(event_id="2EFMS4JI349", season=5)
    s5_valencia2022 = HyroxEvent(event_id="2EFMS4JI371", season=5)
    s5_amsterdam2022 = HyroxEvent(event_id="2EFMS4JI385", season=5)
    s5_chicago2022 = HyroxEvent(event_id="2EFMS4JI39A", season=5)
    s5_berlin2022 = HyroxEvent(event_id="2EFMS4JI399", season=5)
    s5_essen2022 = HyroxEvent(event_id="JGDMS4JI3D5", season=5)
    s5_london2022 = HyroxEvent(event_id="2EFMS4JI3AD", season=5)
    s5_dallas2022 = HyroxEvent(event_id="JGDMS4JI3D7", season=5)
    s5_hongkong2022 = HyroxEvent(event_id="2EFMS4JI3C1", season=5)
    s5_hamburg2022 = HyroxEvent(event_id="JGDMS4JI3E9", season=5)
    s5_madrid2022 = HyroxEvent(event_id="JGDMS4JI3D6", season=5)
    s5_frankfurt2022 = HyroxEvent(event_id="JGDMS4JI3FD", season=5)
    s5_losangeles2022 = HyroxEvent(event_id="JGDMS4JI3FE", season=5)
    s5_maastricht2023 = HyroxEvent(event_id="JGDMS4JI411", season=5)
    s5_manchester2023 = HyroxEvent(event_id="JGDMS4JI425", season=5)
    s5_stuttgart2023 = HyroxEvent(event_id="JGDMS4JI44D", season=5)
    s5_bilbao2023 = HyroxEvent(event_id="JGDMS4JI44F", season=5)
    s5_chicago2023 = HyroxEvent(event_id="JGDMS4JI44E", season=5)
    s5_glasgow2023 = HyroxEvent(event_id="JGDMS4JI439", season=5)
    s5_houston2023 = HyroxEvent(event_id="JGDMS4JI462", season=5) 
    s5_wien2023 = HyroxEvent(event_id="JGDMS4JI461", season=5)
    s5_karlsruhe2023 = HyroxEvent(event_id="JGDMS4JI465", season=5)
    s5_miami2023 = HyroxEvent(event_id="JGDMS4JI474", season=5)
    s5_malaga2023 = HyroxEvent(event_id="JGDMS4JI46A", season=5)
    s5_koln2023 = HyroxEvent(event_id="JGDMS4JI468", season=5)
    s5_anaheim2023 = HyroxEvent(event_id="JGDMS4JI472", season=5)
    s5_hannover2023 = HyroxEvent(event_id="JGDMS4JI46C", season=5)
    s5_rotterdam2023 = HyroxEvent(event_id="JGDMS4JI46E", season=5)
    s5_manchesterWC = HyroxEvent(event_id="2EFMS4JI335", season=5)
    s5_munchen2023 = HyroxEvent(event_id="JGDMS4JI464", season=5)
    s5_hamburg2023 = HyroxEvent(event_id="JGDMS4JI473", season=5)
    s5_barcelona2023 = HyroxEvent(event_id="JGDMS4JI466", season=5)
    s5_dallas2023 = HyroxEvent(event_id="JGDMS4JI470", season=5)
    s5_hongkong2023 = HyroxEvent(event_id="JGDMS4JI46F", season=5)
    s5_london2023 = HyroxEvent(event_id="JGDMS4JI47A", season=5)
    s5_stockholm2023 = HyroxEvent(event_id="JGDMS4JI467", season=5)
    
    s6_rimini2023 = HyroxEvent(event_id="JGDMS4JI4D9", season=6)
    s6_newyork2023 = HyroxEvent(event_id="JGDMS4JI471", season=6)
    s6_sydney2023 = HyroxEvent(event_id="JGDMS4JI515", season=6)
    s6_melbourne2023 = HyroxEvent(event_id="JGDMS4JI516", season=6)
    s6_dubai2023 = HyroxEvent(event_id="JGDMS4JI477", season=6)
    s6_warschau2023 = HyroxEvent(event_id="JGDMS4JI53D", season=6)
    s6_madrid2023 = HyroxEvent(event_id="JGDMS4JI565", season=6)
    s6_malmo2023 = HyroxEvent(event_id="JGDMS4JI46D", season=6)
    s6_milan2023 = HyroxEvent(event_id="JGDMS4JI5A1", season=6)
    s6_singapore2023 = HyroxEvent(event_id="JGDMS4JI58D", season=6)
    s6_munchen2023 = HyroxEvent(event_id="JGDMS4JI5C9", season=6)
    s6_birmingham2023 = HyroxEvent(event_id="JGDMS4JI5E2", season=6)
    s6_valencia2023 = HyroxEvent(event_id="JGDMS4JI5BB", season=6)
    s6_dublin2023 = HyroxEvent(event_id="JGDMS4JI5DD", season=6)
    s6_amsterdam2023 = HyroxEvent(event_id="JGDMS4JI606", season=6)
    s6_paris2023 = HyroxEvent(event_id="JGDMS4JI579", season=6)
    s6_chicago2023 = HyroxEvent(event_id="JGDMS4JI52A", season=6)
    s6_hamburg2023 = HyroxEvent(event_id="JGDMS4JI619", season=6)
    s6_barcelona2023 = HyroxEvent(event_id="JGDMS4JI643", season=6)
    s6_dallas2023 = HyroxEvent(event_id="JGDMS4JI642", season=6)
    s6_hongkong2023 = HyroxEvent(event_id="JGDMS4JI5F2", season=6)
    s6_london2023 = HyroxEvent(event_id="JGDMS4JI62E", season=6)
    s6_stockholm2023 = HyroxEvent(event_id="JGDMS4JI656", season=6)
    s6_frankfurt2023 = HyroxEvent(event_id="JGDMS4JI655", season=6)
    s6_losangeles2023 = HyroxEvent(event_id="JGDMS4JI66A", season=6)

    def extract_event(event: Event, save_dir: str = "/kaggle/working"):
    hyrox_event = event.value.copy()
    hyrox_event.get_info()
    hyrox_event.save(directory=save_dir)

    extract_event(Event.s6_losangeles2023)