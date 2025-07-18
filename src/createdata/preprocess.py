import math

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from src.createdata.preprocess_fighter_data import FighterDetailProcessor

from src.createdata.data_files_path import (  # isort:skip
    FIGHTER_DETAILS,
    PREPROCESSED_DATA,
    TOTAL_EVENT_AND_FIGHTS,
    UFC_DATA,
)

class Preprocessor:
    def __init__(self):
        self.FIGHTER_DETAILS_PATH = FIGHTER_DETAILS
        self.TOTAL_EVENT_AND_FIGHTS_PATH = TOTAL_EVENT_AND_FIGHTS
        self.PREPROCESSED_DATA_PATH = PREPROCESSED_DATA
        self.UFC_DATA_PATH = UFC_DATA
        print("Reading Files")
        self.fights, self.fighter_details = self._read_files()
        self.store = None
        self.processed_fights = None

    def preprocess_fights_for_prediction(self, filtered_fights):
        self.fights = filtered_fights

        print("Renaming Columns")
        self._rename_columns()
        self._replacing_winner_nans_draw()

        print("Converting Percentages to Fractions")
        self._convert_percentages_to_fractions()
        self.fights = self.create_title_bout_feature(self.fights)
        self._create_weight_classes()
        self._convert_last_round_to_seconds()
        self._convert_CTRL_to_seconds()
        self._get_total_time_fought()
        self.store = self._store_compiled_fighter_data_in_another_DF()
        self._create_winner_feature()
        self.processed_fights = self._one_hot_encode_win()

        return self.processed_fights


    def process_raw_data(self, ewm_span=4, ewm_adjust=False):
        """ """

        # Process Fights DF
        print("Renaming Columns")
        self._rename_columns()
        self._replacing_winner_nans_draw()
        print("Converting Percentages to Fractions")
        self._convert_percentages_to_fractions()
        self.fights = self.create_title_bout_feature(self.fights)
        self._create_weight_classes()
        self._convert_last_round_to_seconds()
        self._convert_CTRL_to_seconds()
        self._get_total_time_fought()
        self.store = self._store_compiled_fighter_data_in_another_DF()
        self._create_winner_feature()
        self.processed_fights = self._one_hot_encode_win()

        # Process Fighter Details DF
        self.fighter_details_processor = FighterDetailProcessor(self.processed_fights, self.fighter_details) # Initialise fighter details processor
        self._process_fighter_details() # Clean the fighter details df

        # Generate new columns and merge DataFrames
        self._create_and_add_fighter_attributes(ewm_span=ewm_span, ewm_adjust=ewm_adjust)
        # Creating Age Column (dependent on both DFs)
        self.store = self.create_fighter_age(self.store)
        self._save(self.store, filepath=self.UFC_DATA_PATH)

        print("Fill NaNs")
        self.store = self._fill_nas(self.store) # operates on self.store
        self.store = self._drop_non_essential_cols(self.store) # operates on self.store
        self._save(self.store, filepath=self.PREPROCESSED_DATA_PATH)
        print("Successfully preprocessed and saved ufc data!\n")

    def _read_files(self):
        try:
            fights_df = pd.read_csv(self.TOTAL_EVENT_AND_FIGHTS_PATH, sep=";")

        except Exception as e:
            raise FileNotFoundError("Cannot find the data/total_fight_data.csv")

        try:
            fighter_details_df = pd.read_csv(
                self.FIGHTER_DETAILS_PATH, index_col="fighter_name"
            )

        except Exception as e:
            raise FileNotFoundError("Cannot find the data/fighter_details.csv")

        return fights_df, fighter_details_df



    def _rename_columns(self):
        columns = [
            "R_SIG_STR.",
            "B_SIG_STR.",
            "R_TOTAL_STR.",
            "B_TOTAL_STR.",
            "R_TD",
            "B_TD",
            "R_HEAD",
            "B_HEAD",
            "R_BODY",
            "B_BODY",
            "R_LEG",
            "B_LEG",
            "R_DISTANCE",
            "B_DISTANCE",
            "R_CLINCH",
            "B_CLINCH",
            "R_GROUND",
            "B_GROUND",
        ]

        attempt_suffix = "_att"
        landed_suffix = "_landed"

        for column in columns:
            self.fights[column + attempt_suffix] = self.fights[column].apply(
                lambda X: int(X.split("of")[1])
            )
            self.fights[column + landed_suffix] = self.fights[column].apply(
                lambda X: int(X.split("of")[0])
            )

        self.fights.drop(columns, axis=1, inplace=True)

    def _replacing_winner_nans_draw(self):
        self.fights["Winner"] = self.fights["Winner"].fillna("Draw")

    def _convert_percentages_to_fractions(self):
        pct_columns = ["R_SIG_STR_pct", "B_SIG_STR_pct", "R_TD_pct", "B_TD_pct"]

        def pct_to_frac(X):
            if X != "---":
                return float(X.replace("%", "")) / 100
            else:
                # if '---' means it's taking pct of `0 of 0`.
                # Taking a call here to consider 0 landed of 0 attempted as 0 percentage
                return 0

        for column in pct_columns:
            self.fights[column] = self.fights[column].apply(pct_to_frac)

    def create_title_bout_feature(self, df):
        df["title_bout"] = df["Fight_type"].apply(
            lambda X: True if "Title Bout" in X else False
        )

        return df

    def _create_weight_classes(self):
        def make_weight_class(X):
            weight_classes = [
                "Women's Strawweight",
                "Women's Bantamweight",
                "Women's Featherweight",
                "Women's Flyweight",
                "Lightweight",
                "Welterweight",
                "Middleweight",
                "Light Heavyweight",
                "Heavyweight",
                "Featherweight",
                "Bantamweight",
                "Flyweight",
                "Open Weight",
            ]

            for weight_class in weight_classes:
                if weight_class in X:
                    return weight_class

            if X == "Catch Weight Bout" or "Catchweight Bout":
                return "Catch Weight"
            else:
                return "Open Weight"

        self.fights["weight_class"] = self.fights["Fight_type"].apply(make_weight_class)

        renamed_weight_classes = {
            "Flyweight": "Flyweight",
            "Bantamweight": "Bantamweight",
            "Featherweight": "Featherweight",
            "Lightweight": "Lightweight",
            "Welterweight": "Welterweight",
            "Middleweight": "Middleweight",
            "Light Heavyweight": "LightHeavyweight",
            "Heavyweight": "Heavyweight",
            "Women's Strawweight": "WomenStrawweight",
            "Women's Flyweight": "WomenFlyweight",
            "Women's Bantamweight": "WomenBantamweight",
            "Women's Featherweight": "WomenFeatherweight",
            "Catch Weight": "CatchWeight",
            "Open Weight": "OpenWeight",
        }

        self.fights["weight_class"] = self.fights["weight_class"].apply(
            lambda weight: renamed_weight_classes[weight]
        )

    def _convert_last_round_to_seconds(self):
        # Converting to seconds
        self.fights["last_round_time"] = self.fights["last_round_time"].apply(
            lambda X: int(X.split(":")[0]) * 60 + int(X.split(":")[1])
        )

    def _convert_CTRL_to_seconds(self):
        # Converting to seconds
        CTRL_columns = ["R_CTRL", "B_CTRL"]

        def conv_to_sec(X):
            if X != "--":
                return int(X.split(":")[0]) * 60 + int(X.split(":")[1])
            else:
                # if '--' means there was no time spent on the ground.
                # Taking a call here to consider this as 0 seconds
                return 0

        for column in CTRL_columns:
            self.fights[column + "_time(seconds)"] = self.fights[column].apply(
                conv_to_sec
            )

        # drop original columns
        self.fights.drop(["R_CTRL", "B_CTRL"], axis=1, inplace=True)

    def _get_total_time_fought(self):
        # '1 Rnd + 2OT (15-3-3)' and '1 Rnd + 2OT (24-3-3)' is not included because it has 3 uneven timed rounds.
        # We'll have to deal with it separately
        time_in_first_round = {
            "3 Rnd (5-5-5)": 5 * 60,
            "5 Rnd (5-5-5-5-5)": 5 * 60,
            "1 Rnd + OT (12-3)": 12 * 60,
            "No Time Limit": 1,
            "3 Rnd + OT (5-5-5-5)": 5 * 60,
            "1 Rnd (20)": 1 * 20,
            "2 Rnd (5-5)": 5 * 60,
            "1 Rnd (15)": 15 * 60,
            "1 Rnd (10)": 10 * 60,
            "1 Rnd (12)": 12 * 60,
            "1 Rnd + OT (30-5)": 30 * 60,
            "1 Rnd (18)": 18 * 60,
            "1 Rnd + OT (15-3)": 15 * 60,
            "1 Rnd (30)": 30 * 60,
            "1 Rnd + OT (31-5)": 31 * 5,
            "1 Rnd + OT (27-3)": 27 * 60,
            "1 Rnd + OT (30-3)": 30 * 60,
        }

        exception_format_time = {
            "1 Rnd + 2OT (15-3-3)": [15 * 60, 3 * 60],
            "1 Rnd + 2OT (24-3-3)": [24 * 60, 3 * 60],
        }

        def get_total_time(row):
            if row["Format"] in time_in_first_round.keys():
                return (row["last_round"] - 1) * time_in_first_round[
                    row["Format"]
                ] + row["last_round_time"]

            elif row["Format"] in exception_format_time.keys():

                if (row["last_round"] - 1) >= 2:
                    return (
                        exception_format_time[row["Format"]][0]
                        + (row["last_round"] - 2)
                        * exception_format_time[row["Format"]][1]
                        + row["last_round_time"]
                    )
                else:
                    return (row["last_round"] - 1) * exception_format_time[
                        row["Format"]
                    ][0] + row["last_round_time"]

        self.fights["total_time_fought(seconds)"] = self.fights.apply(
            get_total_time, axis=1
        )
        self.fights.drop(
            ["Format", "last_round_time"], axis=1, inplace=True
        )
        self.fights.drop(["Fight_type"], axis=1, inplace=True)

    def _store_compiled_fighter_data_in_another_DF(self):
        store = self.fights.copy()
        store.drop(
            [
                "R_KD",
                "B_KD",
                "R_SIG_STR_pct",
                "B_SIG_STR_pct",
                "R_TD_pct",
                "B_TD_pct",
                "R_SUB_ATT",
                "B_SUB_ATT",
                "R_REV",
                "B_REV",
                "R_CTRL_time(seconds)",
                "B_CTRL_time(seconds)",
                "win_by",
                "last_round",
                "R_SIG_STR._att",
                "R_SIG_STR._landed",
                "B_SIG_STR._att",
                "B_SIG_STR._landed",
                "R_TOTAL_STR._att",
                "R_TOTAL_STR._landed",
                "B_TOTAL_STR._att",
                "B_TOTAL_STR._landed",
                "R_TD_att",
                "R_TD_landed",
                "B_TD_att",
                "B_TD_landed",
                "R_HEAD_att",
                "R_HEAD_landed",
                "B_HEAD_att",
                "B_HEAD_landed",
                "R_BODY_att",
                "R_BODY_landed",
                "B_BODY_att",
                "B_BODY_landed",
                "R_LEG_att",
                "R_LEG_landed",
                "B_LEG_att",
                "B_LEG_landed",
                "R_DISTANCE_att",
                "R_DISTANCE_landed",
                "B_DISTANCE_att",
                "B_DISTANCE_landed",
                "R_CLINCH_att",
                "R_CLINCH_landed",
                "B_CLINCH_att",
                "B_CLINCH_landed",
                "R_GROUND_att",
                "R_GROUND_landed",
                "B_GROUND_att",
                "B_GROUND_landed",
                "total_time_fought(seconds)",
            ],
            axis=1,
            inplace=True,
        )
        return store

    def _create_winner_feature(self):
        def get_renamed_winner(row):
            r_fighter = row["R_fighter"]
            b_fighter = row["B_fighter"]
            winner = row["Winner"]

            if winner == "Draw":
                return "Draw"

            # Exact match check first
            if winner == r_fighter:
                return "Red"
            if winner == b_fighter:
                return "Blue"

            # Fuzzy matching
            def similarity(a, b):
                return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

            sim_r = similarity(winner, r_fighter)
            sim_b = similarity(winner, b_fighter)

            threshold = 0.8

            if sim_r >= threshold and sim_r >= sim_b:
                print(f"[INFO] Fuzzy match accepted: Winner '{winner}' ≈ R_fighter '{r_fighter}' (sim={sim_r:.2f})")
                return "Red"

            if sim_b >= threshold and sim_b > sim_r:
                print(f"[INFO] Fuzzy match accepted: Winner '{winner}' ≈ B_fighter '{b_fighter}' (sim={sim_b:.2f})")
                return "Blue"

            # If not resolved
            raise ValueError(f"Unable to resolve winner for: {row}")

        self.store["Winner"] = self.store[["R_fighter", "B_fighter", "Winner"]].apply(
            get_renamed_winner, axis=1
        )
    def _process_fighter_details(self):
        self.fighter_details_processor.clean_fighter_details()

    def _create_and_add_fighter_attributes(self, ewm_span=4, ewm_adjust=False):
        self.fighter_details_processor.generate_fighter_stats_columns(ewm_span, ewm_adjust)
        frame = self.fighter_details_processor.frame
        self.store = self.store.join(frame, how="outer")

    def create_fighter_age(self, df):
        df["R_DOB"] = pd.to_datetime(df["R_DOB"])
        df["B_DOB"] = pd.to_datetime(df["B_DOB"])
        df["date"] = pd.to_datetime(df["date"])

        def get_age(row):
            B_age = (row["date"] - row["B_DOB"]).days
            R_age = (row["date"] - row["R_DOB"]).days

            if np.isnan(B_age) != True:
                B_age = math.floor(B_age / 365.25)

            if np.isnan(R_age) != True:
                R_age = math.floor(R_age / 365.25)

            return pd.Series([B_age, R_age], index=["B_age", "R_age"])

        df[["B_age", "R_age"]] = df[["date", "R_DOB", "B_DOB"]].apply(
            get_age, axis=1
        )
        df.drop(["R_DOB", "B_DOB"], axis=1, inplace=True) #Drop DOB now we have age.
        return df

    def _save(self, df, filepath):
        df.to_csv(filepath, index=False)

    def _fill_nas(self, df):
        #self.store["R_Reach_cms"].fillna(self.store["R_Height_cms"], inplace=True)
        #self.store["B_Reach_cms"].fillna(self.store["B_Height_cms"], inplace=True)
        df["R_Reach_cms"] = df["R_Reach_cms"].fillna(df["R_Height_cms"])
        df["B_Reach_cms"] = df["B_Reach_cms"].fillna(df["B_Height_cms"])
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        #self.store.fillna(self.store.median(), inplace=True)

        df["R_Stance"] = df["R_Stance"].fillna("Orthodox")
        df["B_Stance"] = df["B_Stance"].fillna("Orthodox")
        return df

    def _drop_non_essential_cols(self, df):

        if "Winner" in df.columns:
            df.drop(df.index[df["Winner"] == "Draw"], inplace=True)

        # Force all stance categories before one-hot encoding
        all_stances = ["Orthodox", "Southpaw", "Switch", "Sideways", "Open Stance"]
        for col in ["B_Stance", "R_Stance"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=all_stances)

        # Force all weight class categories before one-hot encoding
        all_weight_classes = [
            "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
            "Welterweight", "Middleweight", "LightHeavyweight", "Heavyweight",
            "WomenStrawweight", "WomenFlyweight", "WomenBantamweight", "WomenFeatherweight",
            "CatchWeight", "OpenWeight"
        ]
        if "weight_class" in df.columns:
            df["weight_class"] = pd.Categorical(df["weight_class"], categories=all_weight_classes)

        # One-hot encode ensuring *all possible categories* are included
        cols_to_encode = [col for col in ["weight_class", "B_Stance", "R_Stance"] if col in df.columns]
        if cols_to_encode:
            df = pd.concat([df, pd.get_dummies(df[cols_to_encode])], axis=1)

        cols_to_drop = [
            "weight_class",
            "B_Stance",
            "R_Stance",
            "Referee",
            "location",
            "date",
            "R_fighter",
            "B_fighter",
        ]
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        return df

    def _one_hot_encode_win(self):
        self.fights = pd.concat(
            [self.fights, pd.get_dummies(self.fights["win_by"], prefix="win_by")],
            axis=1,
        )
        self.fights.drop(["win_by"], axis=1, inplace=True)
        return self.fights
