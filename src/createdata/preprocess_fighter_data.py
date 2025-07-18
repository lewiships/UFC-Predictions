import re

import numpy as np
import pandas as pd
from tqdm import tqdm


class FighterDetailProcessor:
    def __init__(self, fights, fighter_details):
        self.fights = fights
        self.fighter_details = fighter_details

        self._initialise_structs()

        self.ewm_span = None
        self.ewm_adjust = None
        self.temp_red_frame, self.temp_blue_frame = None, None
        self.frame = None

    def _initialise_structs(self):
        self.numerical_columns = [
            "hero_KD",
            "opp_KD",
            "hero_SIG_STR_pct",
            "opp_SIG_STR_pct",
            "hero_TD_pct",
            "opp_TD_pct",
            "hero_SUB_ATT",
            "opp_SUB_ATT",
            "hero_REV",
            "opp_REV",
            "hero_SIG_STR._att",
            "hero_SIG_STR._landed",
            "opp_SIG_STR._att",
            "opp_SIG_STR._landed",
            "hero_TOTAL_STR._att",
            "hero_TOTAL_STR._landed",
            "opp_TOTAL_STR._att",
            "opp_TOTAL_STR._landed",
            "hero_TD_att",
            "hero_TD_landed",
            "opp_TD_att",
            "opp_TD_landed",
            "hero_HEAD_att",
            "hero_HEAD_landed",
            "opp_HEAD_att",
            "opp_HEAD_landed",
            "hero_BODY_att",
            "hero_BODY_landed",
            "opp_BODY_att",
            "opp_BODY_landed",
            "hero_LEG_att",
            "hero_LEG_landed",
            "opp_LEG_att",
            "opp_LEG_landed",
            "hero_DISTANCE_att",
            "hero_DISTANCE_landed",
            "opp_DISTANCE_att",
            "opp_DISTANCE_landed",
            "hero_CLINCH_att",
            "hero_CLINCH_landed",
            "opp_CLINCH_att",
            "opp_CLINCH_landed",
            "hero_GROUND_att",
            "hero_GROUND_landed",
            "opp_GROUND_att",
            "opp_GROUND_landed",
            "hero_CTRL_time(seconds)",
            "opp_CTRL_time(seconds)",
            "total_time_fought(seconds)",
        ]
        self.result_stats = [
            "current_win_streak",
            "current_lose_streak",
            "longest_win_streak",
            "wins",
            "losses",
            "draw",
        ]

        self.win_by_columns = [
            "win_by_Decision - Majority",
            "win_by_Decision - Split",
            "win_by_Decision - Unanimous",
            "win_by_KO/TKO",
            "win_by_Submission",
            "win_by_TKO - Doctor's Stoppage",
        ]

    def clean_fighter_details(self):
        # Clean fighter details
        self._drop_future_fighter_details_columns()
        self._convert_height_reach_to_cms()
        self._convert_weight_to_pounds()

    def generate_fighter_stats_columns(self, ewm_span, ewm_adjust):
        self.ewm_span = ewm_span
        self.ewm_adjust = ewm_adjust
        self.temp_red_frame, self.temp_blue_frame = self._calculate_fighter_data()
        self.frame = self.merge_frames(self.fighter_details, self.temp_red_frame, self.temp_blue_frame)
        self.frame = self.rename_columns(self.frame)

    def _drop_future_fighter_details_columns(self):
        self.fighter_details.drop(
            columns=[
                "SLpM",
                "Str_Acc",
                "SApM",
                "Str_Def",
                "TD_Avg",
                "TD_Acc",
                "TD_Def",
                "Sub_Avg",
            ],
            inplace=True,
        )

    def _get_fighters(self):
        # Returns a list of unique fighters.

        red_fighters = self.fights["R_fighter"].value_counts().index
        blue_fighters = self.fights["B_fighter"].value_counts().index

        return list(set(red_fighters) | set(blue_fighters))

    def _calculate_fighter_data(self):
        # Initialising empty data frames
        temp_blue_frame = pd.DataFrame()
        temp_red_frame = pd.DataFrame()

        fighters = self._get_fighters()

        print("Creating Fighter Level Features")
        for fighter_name in tqdm(fighters): # tqdm is showing progress bar. Loop once for each unique fighter.
            # Gives all rows for this fighter back with hero/opp notation.
            # vv Both will be filled if fighter has fights in both corners. vv
            fighter_red = self._get_fighter_red(fighter_name)
            fighter_blue = self._get_fighter_blue(fighter_name)
            fighter_index = None

            if fighter_red is None:
                fighter = fighter_blue
                fighter_index = "blue"
            elif fighter_blue is None:
                fighter = fighter_red
                fighter_index = "red"
            else:
                fighter = pd.concat([fighter_red, fighter_blue]).sort_index()

            fighter["Winner"] = fighter["Winner"].apply(
                lambda X: "hero" if X == fighter_name else "opp"
            )
            # fighter is a df containing all rows where a single fighter has fought,
            # With columns renamed to hero/opp notation and sorted by date.

            for i, index in enumerate(fighter.index):
            # for each fight a fighter fought.

                fighter_slice = fighter[(i + 1) :].sort_index(ascending=False)
                #print(list(fighter_slice))
                # fighter slice is all previous fights.
                # EWM gives more weighting to recent fights.
                s = (
                    fighter_slice[self.numerical_columns]
                    .ewm(span=self.ewm_span, adjust=self.ewm_adjust)
                    .mean()
                    .tail(1)
                )
                if len(s) != 0:
                    pass
                else: # If there are no previous fights, wack in NaN
                    s.loc[len(s)] = [np.nan for _ in s.columns]
                # Adding some more columns to s
                s["total_rounds_fought"] = fighter_slice["last_round"].sum()
                s["total_title_bouts"] = fighter_slice[
                    fighter_slice["title_bout"] == True
                ]["title_bout"].count()
                s["hero_fighter"] = fighter_name
                results = self._get_result_stats(list(fighter_slice["Winner"]))
                for result_stat, result in zip(self.result_stats, results):
                    s[result_stat] = result # add the results data columns to s

                # Compute how many times of each win-type.
                # Add the sum of each win-type thus far to the df.
                win_by_results = fighter_slice[fighter_slice["Winner"] == "hero"][
                    self.win_by_columns
                ].sum()
                for win_by_column, win_by_result in zip(self.win_by_columns, win_by_results):
                    s[win_by_column] = win_by_result

                s.index = [index] # set index of s to match the original fight row index

                if fighter_index is None:
                    if index in fighter_blue.index: # If fighter was blue this time.
                        temp_blue_frame = pd.concat([temp_blue_frame, s])
                    elif index in fighter_red.index: # If fighter was red this time.
                        temp_red_frame = pd.concat([temp_red_frame, s])
                elif fighter_index == "blue": # All blue fights
                    temp_blue_frame = pd.concat([temp_blue_frame, s])
                elif fighter_index == "red": # All red fights.
                    temp_red_frame = pd.concat([temp_red_frame, s])

        """Each temp frame contains all fights from the perspective of the relevant corner."""
        return temp_red_frame, temp_blue_frame

    def generate_fighter_stats_for_prediction(self, fighter_name, fight_date, corner, ewm_span=4, ewm_adjust=False):
        """Generate a synthetic stats row for a fighter for prediction purposes."""
        self.ewm_span = ewm_span
        self.ewm_adjust = ewm_adjust
        # Get all fights for this fighter
        fighter_df_red = self._get_fighter_red(fighter_name)
        fighter_df_blue = self._get_fighter_blue(fighter_name)

        if fighter_df_red is None and fighter_df_blue is None:
            fighter_df = None
        elif fighter_df_red is None:
            fighter_df = fighter_df_blue
        elif fighter_df_blue is None:
            fighter_df = fighter_df_red
        else:
            fighter_df = pd.concat([fighter_df_red, fighter_df_blue]).sort_index()

        # Calculate and print number of fights the fighter has in your dataset
        # num_fights = 0 if fighter_df is None else len(fighter_df)
        # print(f"[INFO] Fighter '{fighter_name}' has {num_fights} UFC fights in dataset before prediction generation.")

        if fighter_df is None or fighter_df.empty:
            print(f"Fighter: {fighter_name} has no fighter data")
            return None

        else:
            fighter_df["Winner"] = fighter_df["Winner"].apply(
                lambda X: "hero" if X == fighter_name else "opp"
            )
            #print(f"Fighter: {fighter_name} has {len(fighter_df)} fighter data")
            # Sort by date descending
            fighter_df = fighter_df.sort_index(ascending=False)

            # Use EWM and aggregates on all fights up to now
            s = fighter_df[self.numerical_columns].ewm(span=self.ewm_span, adjust=self.ewm_adjust).mean().tail(1).iloc[0]

            # Add additional aggregates
            s["total_rounds_fought"] = fighter_df["last_round"].sum()
            s["total_title_bouts"] = fighter_df[fighter_df["title_bout"] == True]["title_bout"].count()
            s["hero_fighter"] = fighter_name

            # Add win streaks and result stats
            results = self._get_result_stats(list(fighter_df["Winner"]))
            #print(f"results: {results}")
            for stat, val in zip(self.result_stats, results):
                s[stat] = val

            win_by_results = fighter_df[fighter_df["Winner"] == "hero"][self.win_by_columns].sum()
            #print(f"win-by results = {win_by_results}")
            for col, val in zip(self.win_by_columns, win_by_results):
                s[col] = val
        #print(f"fighter stats generated for {fighter_name}: {s}")
        #print(f"data for {fighter_name} new data:", s[self.numerical_columns])
        # s is a new row of generated stats for a single fighter up to the present.
        return s

    @staticmethod
    def lreplace(pattern, sub, string):
        """
        Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
        """
        return re.sub("^%s" % pattern, sub, string)

    def _get_fighter_red(self, fighter_name):

        fighter_red = self.fights[self.fights["R_fighter"].str.strip() == fighter_name.strip()]

        if fighter_red.empty:
            return None

        rename_columns = {}
        for column in fighter_red.columns:

            if re.search("^R_", column) is not None:
                rename_columns[column] = self.lreplace("R_", "hero_", column)

            elif re.search("^B_", column) is not None:
                rename_columns[column] = self.lreplace("B_", "opp_", column)

        fighter_red = fighter_red.rename(rename_columns, axis="columns")
        return fighter_red

    def _get_fighter_blue(self, fighter_name):
        fighter_blue = self.fights[self.fights["B_fighter"].str.strip() == fighter_name.strip()]

        if fighter_blue.empty:
            return None

        rename_columns = {}
        for column in fighter_blue.columns:

            if re.search("^B_", column) is not None:
                rename_columns[column] = self.lreplace("B_", "hero_", column)

            elif re.search("^R_", column) is not None:
                rename_columns[column] = self.lreplace("R_", "opp_", column)

        fighter_blue = fighter_blue.rename(rename_columns, axis="columns")
        return fighter_blue

    @staticmethod
    def _get_result_stats(result_list):
        result_list.reverse()  # To get it in ascending order
        current_win_streak = 0
        current_lose_streak = 0
        longest_win_streak = 0
        wins = 0
        losses = 0
        draw = 0
        for result in result_list:

            if result == "hero":
                wins += 1
                current_win_streak += 1
                current_lose_streak = 0
                if longest_win_streak < current_win_streak:
                    longest_win_streak = current_win_streak

            elif result == "opp":
                losses += 1
                current_win_streak = 0
                current_lose_streak += 1

            elif result == "draw":
                draw += 1
                current_lose_streak = 0
                current_win_streak = 0

        return (
            current_win_streak,
            current_lose_streak,
            longest_win_streak,
            wins,
            losses,
            draw,
        )

    def _convert_height_reach_to_cms(self):
        def convert_to_cms(X):

            if X is np.nan:
                return X

            elif len(X.split("'")) == 2:
                feet = float(X.split("'")[0])
                inches = int(X.split("'")[1].replace(" ", "").replace('"', ""))
                return (feet * 30.48) + (inches * 2.54)

            else:
                return float(X.replace('"', "")) * 2.54

        self.fighter_details["Height_cms"] = self.fighter_details["Height"].apply(
            convert_to_cms
        )
        self.fighter_details["Reach_cms"] = self.fighter_details["Reach"].apply(
            convert_to_cms
        )

    def _convert_weight_to_pounds(self):
        self.fighter_details["Weight_lbs"] = self.fighter_details["Weight"].apply(
            lambda X: float(X.replace(" lbs.", "")) if X is not np.nan else X
        )
        self.fighter_details.drop(["Height", "Weight", "Reach"], axis=1, inplace=True)

    def merge_frames(self, fighter_details, red_frame, blue_frame):

        fighter_details.reset_index(inplace=True)
        red_frame.reset_index(inplace=True)
        blue_frame.reset_index(inplace=True)

        blue_frame = blue_frame.merge(
            fighter_details,
            left_on="hero_fighter",
            right_on="fighter_name",
            how="left",
        )
        blue_frame.set_index("index", inplace=True)
        #print(f"blue_frame: {blue_frame.head()} NaN? : {blue_frame.isna().any().sum()}")

        red_frame = red_frame.merge(
            fighter_details,
            left_on="hero_fighter",
            right_on="fighter_name",
            how="left",
        )
        red_frame.set_index("index", inplace=True)
        #print(f"red frame: {red_frame.head()} NaN? : {red_frame.isna().any().sum()}")

        # Dropping redundnat fighter nanme column, same as hero fighter.
        blue_frame.drop("fighter_name", axis=1, inplace=True)
        red_frame.drop("fighter_name", axis=1, inplace=True)

        # Add prefixes so we know which data came from where.
        blue_frame = blue_frame.add_prefix("B_")
        red_frame = red_frame.add_prefix("R_")

        if len(blue_frame) == 1 and len(red_frame) == 1:
            # For predict_fight: merge single rows into a single row
            new_frame = pd.concat([blue_frame.reset_index(drop=True), red_frame.reset_index(drop=True)], axis=1)
            print("[INFO] Using concat fallback for single-row frames with non-matching indices.")
        else:
            # Normal pipeline: indices align
            new_frame = blue_frame.join(red_frame, how="outer")
            if blue_frame.index.equals(red_frame.index):
                # print("[INFO] Using index-based join as indices match.")
                raise ValueError(
                    f"Cannot merge: indices do not match and frame lengths are not 1.\n"
                    f"blue_frame.index={blue_frame.index}, red_frame.index={red_frame.index}, "
                    f"blue_frame.shape={blue_frame.shape}, red_frame.shape={red_frame.shape}"
                )

        #print(f"Merged frame. Shape = {new_frame.shape}, head:\n{new_frame.head()}")

        return new_frame

    def merge_rows(self, fighter_details, red_row, blue_row):
        """
        Merge single-row red and blue DataFrames with fighter_details
        for single-fight prediction, preserving prefixing and structure
        equivalent to merge_frames but without index pollution.
        """

        # Defensive copy to avoid in-place modification
        red_row = red_row.copy()
        blue_row = blue_row.copy()

        # Merge with fighter details
        blue_merged = blue_row.merge(
            fighter_details,
            left_on="hero_fighter",
            right_on="fighter_name",
            how="left",
        )

        red_merged = red_row.merge(
            fighter_details,
            left_on="hero_fighter",
            right_on="fighter_name",
            how="left",
        )
        # print("Red fighter columns:")
        # print(list(red_merged.columns))
        #
        # print("Blue fighter columns:")
        # print(list(blue_merged.columns))
        #
        # print(f"Blue merged fighter name = {blue_merged["hero_fighter"].iloc[0]}")

        # # Defensive: Check if fighter details merged successfully
        # if "fighter_name" not in blue_merged.columns or blue_merged["fighter_name"].isna().all():
        #     raise ValueError(
        #         f"Fighter {blue_merged['fighter_name'].iloc[0]} not found in fighter_details_df for Blue corner.")
        # if "fighter_name" not in red_merged.columns or red_merged["fighter_name"].isna().all():
        #     raise ValueError(
        #         f"Fighter {red_merged['fighter_name'].iloc[0]} not found in fighter_details_df for Red corner.")

        # # Drop redundant 'fighter_name' column
        # blue_merged.drop("fighter_name", axis=1, inplace=True)
        # red_merged.drop("fighter_name", axis=1, inplace=True)

        # Add prefixes
        blue_merged = blue_merged.add_prefix("B_")
        red_merged = red_merged.add_prefix("R_")

        # Reset index to ensure clean horizontal concat
        blue_merged.reset_index(drop=True, inplace=True)
        red_merged.reset_index(drop=True, inplace=True)

        # Concatenate horizontally (axis=1) to produce a single-row merged DataFrame
        merged_row = pd.concat([blue_merged, red_merged], axis=1)

        return merged_row

    def rename_columns(self, df):

        rename_cols = {}

        for col in df.columns:
            if "hero" in col:
                rename_cols[col] = col.replace("_hero_", "_avg_").replace(".", "")
            if "opp" in col:
                rename_cols[col] = col.replace("_opp_", "_avg_opp_").replace(".", "")
            if "win_by" in col:
                rename_cols[col] = (
                    col.replace(" ", "").replace("-", "_").replace("'s", "_")
                )

        df.rename(rename_cols, axis="columns", inplace=True)
        df.drop(["R_avg_fighter", "B_avg_fighter"], axis=1, inplace=True)

        return df
