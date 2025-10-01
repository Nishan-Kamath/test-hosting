# train_model_3features.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

import sklearn
print(sklearn.__version__)

# Load dataset
url = "https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv"
df = pd.read_csv('voice.csv')

# Encode label
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Select only 3 features
X = df[["meanfreq", "sd", "median"]]
y = df["label"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Test Accuracy:", model.score(X_test, y_test))

# Save
joblib.dump(model, "gender_model_3features.pkl")
joblib.dump(scaler, "scaler_3features.pkl")
joblib.dump(le, "label_encoder_3features.pkl")
