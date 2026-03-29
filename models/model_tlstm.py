import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
import os

# ------------------------- Configuration -------------------------
WINDOW_SIZE = 10
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ------------------------- Load Data -------------------------
X = np.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/X_train.npy")
y = np.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/y_train.npy")
scaler_y = joblib.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/common_scaler_y.pkl")
feature = joblib.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/common_feature_columns.pkl")

# ------------------------- Sequence Creation -------------------------
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, WINDOW_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Extract time_delta sequences
time_delta_index = feature.index('time_delta')
delta_train = torch.tensor(X_train[:, :, time_delta_index:time_delta_index+1], dtype=torch.float32).to(DEVICE)
delta_val = torch.tensor(X_val[:, :, time_delta_index:time_delta_index+1], dtype=torch.float32).to(DEVICE)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

# ------------------------- T-LSTM Cell -------------------------
class TLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_d = nn.Linear(1, hidden_dim)
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, x, h, c, delta_t):
        gamma = torch.exp(-F.relu(self.W_d(delta_t)))
        c = gamma * c
        h, c = self.lstm_cell(x, (h, c))
        return h, c

# ------------------------- T-LSTM Model -------------------------
class TLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tlstm_cell = TLSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, delta_seq):
        batch_size, seq_len, _ = x_seq.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)

        for t in range(seq_len):
            h, c = self.tlstm_cell(x_seq[:, t], h, c, delta_seq[:, t])
        h = self.dropout(h)
        return self.fc(h)

# ------------------------- Training -------------------------
model = TLSTMPredictor(input_dim=X.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

train_loader = DataLoader(TensorDataset(X_train_tensor, delta_train, y_train_tensor), batch_size=BATCH_SIZE)
val_loader = DataLoader(TensorDataset(X_val_tensor, delta_val, y_val_tensor), batch_size=BATCH_SIZE)

best_val_loss = float('inf')
patience_counter = 0

print("\n🚀 Training T-LSTM with Early Stopping...")
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for xb, tb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb, tb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, tb, yb in val_loader:
            preds = model(xb, tb).squeeze()
            loss = loss_fn(preds, yb)
            val_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "C:/Users/kdhas/Desktop/Research New/models/new models/new/tlstm_traffic_model.pth")
        print(f"✅ Saved best model at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

# ------------------------- Evaluation -------------------------
def evaluate_and_plot(X_tensor, y_tensor, delta_tensor, label, max_plot=500):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, tb, yb in DataLoader(TensorDataset(X_tensor, delta_tensor, y_tensor), batch_size=BATCH_SIZE):
            pred = model(xb, tb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.cpu().numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    preds_unscaled = scaler_y.inverse_transform(preds)
    targets_unscaled = scaler_y.inverse_transform(targets)

    mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    mse = mean_squared_error(targets_unscaled, preds_unscaled)
    r2 = r2_score(targets_unscaled, preds_unscaled)

    print(f"\n📊 {label} Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(targets_unscaled[:max_plot], label='Actual')
    plt.plot(preds_unscaled[:max_plot], label='Predicted')
    plt.title(f"{label} Set: Actual vs Predicted Packet Rate")
    plt.xlabel("Time Step")
    plt.ylabel("Packet Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------- Final Results -------------------------
evaluate_and_plot(X_train_tensor, y_train_tensor, delta_train, "Train")
evaluate_and_plot(X_val_tensor, y_val_tensor, delta_val, "Validation")
