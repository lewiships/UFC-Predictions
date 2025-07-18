import pickle
import sklearn

with open(r"C:\Users\lewis\repos\UFC-Predictions\saved_models\model_24_06_25.sav", "rb") as f:
    model = pickle.load(f)

import numpy as np

# Example: if your model expects 50 features
X_dummy = np.random.rand(1, 176)  # 1 sample, 50 features

prediction = model.predict(X_dummy)
print("Prediction:", prediction)

proba = model.predict_proba(X_dummy)
print("Probability:", proba)