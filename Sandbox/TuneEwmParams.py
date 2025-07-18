""" 08-07-25: Found that a span of 4 with adjust False gave the best model accuracy.
This is a marginal improvement over the original 3 but useful to know."""


import pandas as pd
from src.createdata.preprocess_fighter_data import FighterDetailProcessor
from ModelCreation.TrainModel import FightModelTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from src.createdata.preprocess import Preprocessor

# 1️⃣ Define EWM param grid
ewm_param_grid = {
    "span": [1, 2, 3, 4, 5, 6],
    "adjust": [True, False]  # or [True, False] if you want to test
}

preprocessor = Preprocessor()

# Load the freshly preprocessed, clean fights data
fights_df = pd.read_csv(preprocessor.UFC_DATA_PATH)
fighter_details_df = pd.read_csv(preprocessor.FIGHTER_DETAILS_PATH, index_col="fighter_name")

results = []
print(fights_df.columns)
print(fights_df.head())

# 2️⃣ Iterate over EWM param combinations
for span in ewm_param_grid["span"]:
    for adjust in ewm_param_grid["adjust"]:
        print(f"\n==== Running for span={span}, adjust={adjust} ====")
        preprocessor.process_raw_data(ewm_span=span, ewm_adjust=adjust) # Processes for current ewm values and saves
        # Train models
        trainer = FightModelTrainer(preprocessor.PREPROCESSED_DATA_PATH, target_column="Winner")

        trainer.train_model(LogisticRegression(max_iter=1000), "LogReg")
        trainer.train_model(RandomForestClassifier(n_estimators=300, n_jobs=-1), "RF")
        trainer.train_model(xgb.XGBClassifier(eval_metric="logloss"), "XGB")
        trainer.train_model(lgb.LGBMClassifier(), "LGBM")

        # Retrieve metrics and determine best model
        metrics_df = trainer.get_metrics()
        best_model_name = metrics_df['accuracy'].idxmax()
        best_metrics = metrics_df.loc[best_model_name]

        # Record results
        result_entry = {
            "span": span,
            "adjust": adjust,
            "best_model": best_model_name,
            "accuracy": best_metrics["accuracy"],
            "roc_auc": best_metrics["roc_auc"],
            "f1_score": best_metrics["f1_score"]
        }
        results.append(result_entry)

        # Optionally save the best model
        trainer.save_model(best_model_name, save_dir=f"models/span{span}_adjust{adjust}")

# 3️⃣ Save all results
results_df = pd.DataFrame(results)
results_df.to_csv("ewm_param_tuning_results.csv", index=False)
print("\nTuning completed. Results saved to 'ewm_param_tuning_results.csv'.")
