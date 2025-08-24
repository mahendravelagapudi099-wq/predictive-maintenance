
import pandas as pd
import joblib

# Load latest data
df = pd.read_csv('data/synthetic_data.csv')
latest = df[['temperature', 'vibration', 'pressure']].tail(1)

# Load model
model = joblib.load('models/xgboost_model.pkl')

# Predict
prob = model.predict_proba(latest)[0][1]
if prob > 0.8:
    print(f"⚠️ ALERT: Failure risk is high! (score={prob:.2f})")
else:
    print(f"✅ Normal operation. (score={prob:.2f})")
