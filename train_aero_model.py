import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/mclaren_aero_dataset.csv")

# Inputs and targets
X = df[["wing_angle", "ride_height", "mode"]]
y = df[["Cd", "Cl"]]

# Preprocessing: one-hot encode mode
preprocessor = ColumnTransformer([
    ("mode_ohe", OneHotEncoder(sparse_output=False), ["mode"]),
    ("num", "passthrough", ["wing_angle", "ride_height"])
])

# RandomForest regressor (multi-output)
rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, min_samples_leaf=3, n_jobs=-1)
multi = MultiOutputRegressor(rf)

# Full pipeline
pipeline = Pipeline([
    ("pre", preprocessor),
    ("reg", multi)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2_cd = r2_score(y_test["Cd"], y_pred[:,0])
r2_cl = r2_score(y_test["Cl"], y_pred[:,1])
print(f"MAE: {mae:.6f}, R2 Cd: {r2_cd:.6f}, R2 Cl: {r2_cl:.6f}")

# Save model
dump(pipeline, "data/mclaren_aero_model.pkl")
print("Saved McLaren surrogate model to data/mclaren_aero_model.pkl")
