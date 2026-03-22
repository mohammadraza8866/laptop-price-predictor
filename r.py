import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ===========================
# Load dataset ONLY (no old model!)
# ===========================
df = joblib.load("df_joblib.pkl")

# ===========================
# Features & target
# ===========================
X = df.drop('Price', axis=1)
y = np.log(df['Price'])

# ===========================
# Columns
# ===========================
categorical = ['Company','TypeName','Cpu brand','Gpu brand','os']
numerical = ['Ram','Weight','Touchscreen','Ips','ppi','HDD','SSD']

# ===========================
# Preprocessing (FRESH OBJECT)
# ===========================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ],
    remainder='passthrough'
)

# ===========================
# NEW PIPELINE (IMPORTANT)
# ===========================
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

# ===========================
# Train
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)

# ===========================
# SAVE (overwrite)
# ===========================
joblib.dump(pipe, "pipe_joblib.pkl")

print("✅ CLEAN MODEL CREATED")