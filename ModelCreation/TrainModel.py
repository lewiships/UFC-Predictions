import pandas as pd
import numpy as np
import joblib
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, auc

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt

class FightModelTrainer:
    def __init__(self, data_path, target_column="Winner", test_size=0.2, random_state=42):
        # Updated 06/25, requires significant computation to calibrate.
        self.best_params = {'bootstrap': False,
                             'criterion': 'gini',
                             'max_depth': 50,
                             'max_features': 'sqrt',
                             'min_samples_leaf': 1,
                             'min_samples_split': 2,
                             'n_estimators': 1200}
        self.start = time.time()
        # Load and prepare data
        self.data = pd.read_csv(data_path)
        self.target_column = target_column

        # Split into features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        # Encoding labels
        self.y = self.y.map({"Red": 1, "Blue": 0})

        # Normalise data.
        scaler = StandardScaler()
        self.X = pd.DataFrame(
            scaler.fit_transform(self.X),
            columns=self.X.columns
        )
        # Save scaler
        joblib.dump(scaler, "models/scaler.pkl")

        # Train/validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        # save the feature names order
        joblib.dump(self.X_train.columns.tolist(), "models/feature_names.pkl")

        # Store trained models and their metrics
        self.models = {}
        self.metrics = {}

    def train_model(self, model, model_name):
        print(f"Training {model_name}...")

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        accuracy = accuracy_score(self.y_val, y_pred)
        roc_auc = roc_auc_score(self.y_val, y_proba) if y_proba is not None else None
        f1 = f1_score(self.y_val, y_pred)

        self.models[model_name] = model
        self.metrics[model_name] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(self.y_val, y_pred)
        }

        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}" if roc_auc else f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Completed in {time.time() - self.start:.2f} seconds")
    def save_model(self, model_name, save_dir="models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(self.models[model_name], os.path.join(save_dir, f"{model_name}.pkl"))
        print(f"{model_name} saved to {save_dir}")

    def get_metrics(self):
        return pd.DataFrame(self.metrics).T

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))

        for model_name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_val)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_val, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    trainer = FightModelTrainer(r"C:\Users\lewis\repos\UFC-Predictions\src\data\preprocessed_data.csv", target_column="Winner")

    # Train different models
    trainer.train_model(LogisticRegression(max_iter=1000), "LogisticRegression")
    trainer.train_model(RandomForestClassifier(**trainer.best_params), "RandomForest")
    trainer.train_model(xgb.XGBClassifier(eval_metric="logloss"), "XGBoost")
    trainer.train_model(lgb.LGBMClassifier(), "LightGBM")

    # View metrics
    print(trainer.get_metrics())

    # Save models
    trainer.save_model("LogisticRegression")
    trainer.save_model("RandomForest")
    trainer.save_model("XGBoost")
    trainer.save_model("LightGBM")

    # Plot ROC curves
    trainer.plot_roc_curves()