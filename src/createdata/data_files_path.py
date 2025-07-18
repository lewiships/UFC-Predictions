import os
from pathlib import Path

#BASE_PATH = Path(os.getcwd()) / "data"
BASE_PATH = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_PATH / "ModelCreation" / "models"
DATA_PATH = BASE_PATH /"src" / "data"

EVENT_AND_FIGHT_LINKS_PICKLE = MODEL_PATH / "event_and_fight_links.pickle"
PAST_EVENT_LINKS_PICKLE = MODEL_PATH / "past_event_links.pickle"
PAST_FIGHTER_LINKS_PICKLE = MODEL_PATH / "past_fighter_links.pickle"
SCRAPED_FIGHTER_DATA_DICT_PICKLE = MODEL_PATH / "scraped_fighter_data_dict.pickle"
NEW_EVENT_AND_FIGHTS = DATA_PATH / "new_fight_data.csv"
TOTAL_EVENT_AND_FIGHTS = DATA_PATH / "raw_total_fight_data.csv"
PREPROCESSED_DATA = DATA_PATH / "preprocessed_data.csv"
FIGHTER_DETAILS = DATA_PATH / "raw_fighter_details.csv"
UFC_DATA = DATA_PATH / "data.csv"

