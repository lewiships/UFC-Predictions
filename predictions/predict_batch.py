import pandas as pd
from pathlib import Path
from tabulate import tabulate
import argparse
from src.createdata.data_files_path import DATA_PATH, MODEL_PATH

from predictions.predict_fight import FightPredictor

# Paths

UPCOMING_FIGHTS_PATH = DATA_PATH / "upcoming_fights.csv"
BATCH_PREDICTIONS_PATH = DATA_PATH / "upcoming_predictions.csv"

def predict_batch(plot=True):
    # Initialise the predictor once
    predictor = FightPredictor(
        model_path=r"C:\Users\lewis\repos\UFC-Predictions\ModelCreation\models\LogisticRegression.pkl",
        scaler_path=r"C:\Users\lewis\repos\UFC-Predictions\ModelCreation\models\scaler.pkl"
    )

    upcoming_df = pd.read_csv(UPCOMING_FIGHTS_PATH)

    missing_fighters = []
    for fighter in upcoming_df["R_fighter"].unique().tolist() + upcoming_df["B_fighter"].unique().tolist():

        if fighter not in predictor.fighter_details_df.index:
            missing_fighters.append(fighter)

    print(f"Missing Fighters: {sorted(set(missing_fighters))}")

    results = []

    for _, row in upcoming_df.iterrows():
        fighter_a = row["R_fighter"]
        fighter_b = row["B_fighter"]
        fight_date = row["date"]
        weight_class = row["Fight_type"]
        title_bout = "Title Bout" if "Title" in weight_class else "Non-Title Bout"

        try:
            predictor.validate_arguments(fighter_a, fighter_b, fight_date, weight_class, title_bout)
            winner, confidence = predictor.predict(fighter_a, fighter_b, fight_date, weight_class, title_bout)
            results.append({
                "Red Fighter": fighter_a,
                "Blue Fighter": fighter_b,
                "Predicted Winner": winner,
                "Confidence (%)": f"{confidence * 1:.1f}"
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Skipping {fighter_a} vs {fighter_b} due to error: {e}")

    if not results:
        print("⚠️ No predictions could be made due to missing data or errors.")
        return

    results_df = pd.DataFrame(results)

    if plot:
        print(tabulate(results_df, headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        results_df.to_csv(BATCH_PREDICTIONS_PATH, index=False)
        print(f"\n✅ Predictions saved to {BATCH_PREDICTIONS_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict upcoming UFC fights using FightPredictor.")
    parser.add_argument("--plot", action="store_true", help="Display predictions in a CLI table instead of saving to CSV.")
    args = parser.parse_args()

    predict_batch(plot=args.plot)
