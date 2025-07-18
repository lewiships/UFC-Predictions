
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from difflib import SequenceMatcher

from src.createdata.preprocess import Preprocessor
from src.createdata.preprocess_fighter_data import FighterDetailProcessor
from src.createdata.data_files_path import (
    TOTAL_EVENT_AND_FIGHTS,
    FIGHTER_DETAILS,
    PREPROCESSED_DATA
)


class FightPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize predictor with saved model and scaler."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(r"C:\Users\lewis\repos\UFC-Predictions\ModelCreation\models\feature_names.pkl")

        self.preprocessor = Preprocessor()

        # Load base fight data and fighter details
        self.fights_df = pd.read_csv(TOTAL_EVENT_AND_FIGHTS, sep=";")
        self.fighter_details_df = pd.read_csv(FIGHTER_DETAILS, index_col="fighter_name")

        self.all_fighters_master = set(str(fighter).strip() for fighter in self.fighter_details_df.index)

        #filtered_fights = self._filter_fights_up_to_date(fight_date)
        clean_fights_df = self.preprocessor.preprocess_fights_for_prediction(self.fights_df)
        self.processor = FighterDetailProcessor(clean_fights_df, self.fighter_details_df)
        self.processor.clean_fighter_details()


    @staticmethod
    def _filter_fights_up_to_date(self, df, fight_date: str) -> pd.DataFrame:
        """Filter fights data to only fights before given fight_date."""
        df["date"] = pd.to_datetime(df["date"])
        return df[df["date"] < pd.to_datetime(fight_date)]

    def _generate_prediction_row(self, fighter_a, fighter_b, fight_date, weight_class, title_bout):
        """Generate a single row for prediction."""
        #filtered_fights = self._filter_fights_up_to_date(df, fight_date)
        # Generate synthetic stats
        red_fighter_stats = self.processor.generate_fighter_stats_for_prediction(fighter_a, fight_date, corner="Red")
        blue_fighter_stats = self.processor.generate_fighter_stats_for_prediction(fighter_b, fight_date, corner="Blue")
        if red_fighter_stats is None or blue_fighter_stats is None:
            print(f"[INFO] Skipping prediction: One or both fighters have no UFC fights in dataset.")
            return None
        red_fighter_df = red_fighter_stats.to_frame().T
        blue_fighter_df = blue_fighter_stats.to_frame().T

        new_row = self.processor.merge_rows(
            self.processor.fighter_details.copy(),
            red_fighter_df.copy(),
            blue_fighter_df.copy()
        )
        nan_columns = new_row.columns[new_row.isna().any()].tolist()

        new_row = self.processor.rename_columns(new_row)

        # New row still needs some processing.
        new_row["date"] = pd.to_datetime(fight_date)
        # Adding age and dropping DOBs
        new_row = self.preprocessor.create_fighter_age(new_row)
        new_row = self.preprocessor._fill_nas(new_row)

        new_row["Fight_type"] = title_bout
        new_row["weight_class"] = weight_class
        new_row = self.preprocessor.create_title_bout_feature(new_row)
        new_row.drop(["Fight_type"], axis=1, inplace=True)
        new_row = self.preprocessor._drop_non_essential_cols(new_row)

        # combined_row = pd.concat([red_fighter_stats, blue_fighter_stats]).to_frame().T
        #
        # fighter_ewm_stats = processor.rename_columns(combined_row)

        return new_row

    def validate_arguments(self, fighter_a: str, fighter_b: str, fight_date: str, weight_class: str, title_bout: str):
        # Validate arguments
        all_fighters = self.all_fighters_master
        if not fighter_a in all_fighters or not fighter_b in all_fighters:
            print(f"{fighter_a} --> {fighter_a in all_fighters}, {fighter_b} --> {fighter_b in all_fighters}")
            a_in = fighter_a in all_fighters
            b_in = fighter_b in all_fighters

            def similarity(a, b):
                return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

            for fighter in all_fighters:
                if not a_in:
                    sim_a = similarity(fighter_a, fighter)
                    if sim_a > 0.8:
                        print(f"Did you mean {fighter}? Similarity = {sim_a}")
                        break
                if not b_in:
                    sim_b = similarity(fighter_b, fighter)
                    if sim_b > 0.8:
                        print(f"Did you mean {fighter}? Similarity = {sim_b}")
                        break

            print(
                f"⚠️ Skipping: {fighter_a} or {fighter_b} not found in fighter_details_df. Skipping prediction for this fight.\n")
            return False

    def predict(self, fighter_a: str, fighter_b: str, fight_date: str, weight_class: str, title_bout: str):
        """Predict winner for given fight."""

        X = self._generate_prediction_row(fighter_a, fighter_b, fight_date, weight_class, title_bout)

        if X is None:
            return "Squash", 100

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0  # Add missing with 0
                print(f"ERROR!! Column: {col} not in the data frame. Replaced with zeros for now.")
        X = X[self.feature_names]  # Reorder to match training


        # Scale
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Predict
        pred = self.model.predict(X_scaled_df)[0]
        proba = self.model.predict_proba(X_scaled_df)[0]

        winner = fighter_a if pred == 1 else fighter_b
        winner_prob = proba[pred]

        print(f"✅ Prediction: {winner} with confidence {winner_prob:.2%}")
        return winner, winner_prob




if __name__ == "__main__":
    predictor = FightPredictor(
        model_path=r"C:\Users\lewis\repos\UFC-Predictions\ModelCreation\models\LogisticRegression.pkl",
        scaler_path=r"C:\Users\lewis\repos\UFC-Predictions\ModelCreation\models\scaler.pkl"
    )
    #predictor.predict("Merab Dvalishvili", "Sean O'Malley", "2025-09-01", "Bantamweight", "Title Bout")
    predictor.predict("Ilia Topuria", "Charles Oliveira", "2025-09-01", "Lightweight", "Title Bout")