import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# ------------------------- Configuration -------------------------
DATA_PATH = "C:/Users/kdhas/Desktop/Research New/models/preprocess data"
MODEL_SAVE_PATH = "C:/Users/kdhas/Desktop/Research New/models/new models/new/new_xgboost_model.pkl"
WINDOW_SIZE = 10

# ------------------------- Load Data -------------------------
X = np.load(os.path.join(DATA_PATH, "test_X_train.npy"))
y = np.load(os.path.join(DATA_PATH, "test_y_train.npy"))
scaler_y = joblib.load(os.path.join(DATA_PATH, "test_common_scaler_y.pkl"))

# ------------------------- Sequence Creation and Flattening -------------------------
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

def flatten_sequences(X_seq):
    return X_seq.reshape((X_seq.shape[0], -1))

X_seq, y_seq = create_sequences(X, y, WINDOW_SIZE)
X_seq_flat = flatten_sequences(X_seq)

# ------------------------- Train-Test Split -------------------------
X_train, X_val, y_train, y_val = train_test_split(X_seq_flat, y_seq, test_size=0.2, shuffle=False)

# ------------------------- Train XGBoost Model -------------------------
print("🚀 Training XGBoost Regressor...")
base_xgb = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    verbosity=1
)
xgb_model = MultiOutputRegressor(base_xgb)
xgb_model.fit(X_train, y_train)

# ------------------------- Save Model -------------------------
joblib.dump(xgb_model, MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# ------------------------- Evaluation -------------------------
def evaluate_and_plot(model, X_data, y_data, label, max_plot=500):
    preds = model.predict(X_data)
    preds_unscaled = scaler_y.inverse_transform(preds)
    targets_unscaled = scaler_y.inverse_transform(y_data)

    mae = mean_absolute_error(targets_unscaled, preds_unscaled, multioutput='raw_values')
    mse = mean_squared_error(targets_unscaled, preds_unscaled, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_unscaled, preds_unscaled, multioutput='raw_values')

    print(f"\n📊 {label} Evaluation:")
    print(f"MAE - Packet Rate: {mae[0]:.4f}, Byte Rate: {mae[1]:.4f}")
    print(f"MSE - Packet Rate: {mse[0]:.4f}, Byte Rate: {mse[1]:.4f}")
    print(f"RMSE - Packet Rate: {rmse[0]:.4f}, Byte Rate: {rmse[1]:.4f}")
    print(f"R²   - Packet Rate: {r2[0]:.4f}, Byte Rate: {r2[1]:.4f}")

    # Plot predictions
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(targets_unscaled[:max_plot, 0], label="Actual Packet Rate")
    plt.plot(preds_unscaled[:max_plot, 0], label="Predicted Packet Rate")
    plt.title(f"{label} - Packet Rate")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(targets_unscaled[:max_plot, 1], label="Actual Byte Rate")
    plt.plot(preds_unscaled[:max_plot, 1], label="Predicted Byte Rate")
    plt.title(f"{label} - Byte Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# ------------------------- Final Results -------------------------
evaluate_and_plot(xgb_model, X_train, y_train, "Train")
evaluate_and_plot(xgb_model, X_val, y_val, "Validation")
