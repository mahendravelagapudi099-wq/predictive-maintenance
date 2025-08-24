
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 10000
timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="H")

# Generate synthetic data
data = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': 60 + 10*np.random.randn(n_samples),
    'vibration': 5 + np.random.randn(n_samples),
    'pressure': 100 + 5*np.random.randn(n_samples),
})

# Simulate failures
data['failure'] = 0
data.loc[::2000, 'failure'] = 1
data['time_to_failure'] = data['failure'][::-1].cumsum()[::-1]
data['label'] = (data['time_to_failure'] < 50).astype(int)

# Save to CSV
data.to_csv('data/synthetic_data.csv', index=False)
print("✅ Synthetic data saved to data/synthetic_data.csv")
