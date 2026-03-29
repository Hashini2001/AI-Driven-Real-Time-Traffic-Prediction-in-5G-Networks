import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
csv_path = 'C:/Users/kdhas/Desktop/Research New/Dataset/preprocess_dataset/combined_dataset_original.csv'
df = pd.read_csv(csv_path, low_memory=False)

# -----------------------------
# 2. Timestamp Conversion
# -----------------------------
# Convert Time column to timedelta (assumes HH:MM:SS format)
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time
df['Time'] = pd.to_timedelta(df['Time'].astype(str))

# Add a fixed base date
df['Timestamp'] = pd.to_datetime('2025-01-01') + df['Time']

# Handle duplicates in Timestamp by adding miliseconds
dup_counts = df.groupby('Timestamp').cumcount()
df['Adjusted_Timestamp'] = df['Timestamp'] + pd.to_timedelta(dup_counts, unit='us')

# Use Adjusted_Timestamp for sorting and time index
df = df.sort_values('Adjusted_Timestamp').reset_index(drop=True)
df.set_index('Adjusted_Timestamp', inplace=True)

# -----------------------------
# 3. Feature Engineering
# -----------------------------

# Time delta (for time-aware models like T-LSTM, T-GRU)
df['time_delta'] = df.index.to_series().diff().dt.total_seconds().fillna(0).astype(float)

# Optionally filter large gaps
max_allowed_gap = 60  # seconds
df = df[df['time_delta'] <= max_allowed_gap]

# Categorical Columns Encoding
label_encoders = {}
cat_cols = [
    'Source', 'Destination', 'Protocol', 'type', 'dns_query_type', 'dns_query_domain',
    'stun_response_type', 'rtcp_type', 'raknet_msg_type', 'quic_scid',
    'gquic_cid', 'mapped_ip'
]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('missing').astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Lag Features
for lag in range(1, 6):
    df[f'lag_packet_rate_{lag}'] = df['packet_rate'].shift(lag)
    df[f'lag_byte_rate_{lag}'] = df['byte_rate'].shift(lag)

# Rolling Mean and Std Dev
df['rolling_packet_mean_3'] = df['packet_rate'].shift(1).rolling(window=3).mean()
df['rolling_packet_std_3'] = df['packet_rate'].shift(1).rolling(window=3).std()
df['rolling_mean_byte_3'] = df['byte_rate'].shift(1).rolling(window=3).mean()
df['rolling_std_byte_3'] = df['byte_rate'].shift(1).rolling(window=3).std()

# Drop rows with NaNs caused by shifting/rolling
df.dropna(inplace=True)
df.reset_index(drop=False, inplace=True)  # Optional: keep Timestamp

# -----------------------------
# 4. Split into X and y
# -----------------------------
# Drop datetime column before scaling
drop_cols = ['packet_rate','byte_rate', 'Time', 'Timestamp', 'Adjusted_Timestamp'] if 'Time' in df.columns else ['packet_rate','byte_rate', 'Timestamp', 'Adjusted_Timestamp']
X = df.drop(columns=drop_cols)
y = df[['packet_rate', 'byte_rate']]

# Save column names
feature_columns = X.columns.tolist()
assert not any(X.dtypes == 'datetime64[ns]'), " Datetime columns still present in features!"

# -----------------------------
# 5. Scaling
# -----------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Log-transform y to reduce skewness (safe for zero values using log1p)
y_log = np.log1p(y)

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_log)

# -----------------------------
# 6. Train/Test Split (time-aware)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, shuffle=False
)

# -----------------------------
# 7. Save Outputs
# -----------------------------
#os.makedirs("C:/Users/kdhas/Desktop/Research New/Dataset", exist_ok=True)
os.makedirs("C:/Users/kdhas/Desktop/Research New/models/preprocess data", exist_ok=True)

# -----------------------------
# 8. Save Entire Preprocessed Dataset to CSV
# -----------------------------
#preprocessed_csv_path = "C:/Users/kdhas/Desktop/Research New/Dataset/preprocess_dataset/combined_dataset_preprocessed.csv"
#df.to_csv(preprocessed_csv_path, index=False)

print("✅ Entire preprocessed dataset saved to CSV.")

# Save test sets
np.save("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_X_train.npy", X_train)
np.save("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_y_train.npy", y_train)
np.save("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_X_test.npy", X_test)
np.save("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_y_test.npy", y_test)

# Save scalers, encoders, feature columns
joblib.dump(scaler_X, "C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_common_scaler_X.pkl")
joblib.dump(scaler_y, "C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_common_scaler_y.pkl")
joblib.dump(label_encoders, "C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_common_label_encoders.pkl")
joblib.dump(feature_columns, "C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_common_feature_columns.pkl")

print("✅ Preprocessing complete. Test data and models saved.")

