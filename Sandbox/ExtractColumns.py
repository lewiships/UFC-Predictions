import pandas as pd


data_path = r"C:\Users\lewis\repos\UFC-Predictions\src\data\preprocessed_data.csv"
data = pd.read_csv(data_path)

print(list(data.columns))

