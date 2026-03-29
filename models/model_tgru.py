import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import LayerNorm

# ------------------------- Configuration -------------------------
WINDOW_SIZE = 10
EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ------------------------- Load Data -------------------------
X = np.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_X_train.npy")
y = np.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_y_train.npy")
scaler_y = joblib.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_common_scaler_y.pkl")
feature = joblib.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_common_feature_columns.pkl")

# ------------------------- Sequence Creation -------------------------
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, WINDOW_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Find index of time_delta column
time_delta_index = feature.index('time_delta')

# Extract time_delta sequences from X
delta_train = torch.tensor(X_train[:, :, time_delta_index:time_delta_index+1], dtype=torch.float32).to(DEVICE)
delta_val = torch.tensor(X_val[:, :, time_delta_index:time_delta_index+1], dtype=torch.float32).to(DEVICE)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Move to device
X_train_tensor, y_train_tensor, delta_train = X_train_tensor.to(DEVICE), y_train_tensor.to(DEVICE), delta_train.to(DEVICE)
X_val_tensor, y_val_tensor, delta_val = X_val_tensor.to(DEVICE), y_val_tensor.to(DEVICE), delta_val.to(DEVICE)

# ------------------------- Model Definition -------------------------
class TGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_d = nn.Linear(1, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x, h, delta_t):
        gamma = torch.exp(-F.relu(self.W_d(delta_t)))
        h_decay = gamma * h
        return self.gru_cell(x, h_decay)

class TGRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim = 2, dropout_rate=0.3):
        super().__init__()
        self.tgru_cell = TGRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq, delta_t_seq):
        batch_size, seq_len, _ = x_seq.size()
        h = torch.zeros(batch_size, self.tgru_cell.hidden_dim).to(x_seq.device)
        for t in range(seq_len):
            h = self.tgru_cell(x_seq[:, t], h, delta_t_seq[:, t])
        h = self.dropout(h)
        return self.fc(h)

# ------------------------- Training -------------------------
model = TGRUPredictor(input_dim=X.shape[1], output_dim=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
train_losses_all = []
val_losses_all = []

train_loader = DataLoader(TensorDataset(X_train_tensor, delta_train, y_train_tensor), batch_size=BATCH_SIZE)
val_loader = DataLoader(TensorDataset(X_val_tensor, delta_val, y_val_tensor), batch_size=BATCH_SIZE)

best_val_loss = float('inf')
patience_counter = 0

print("\n🚀 Training with Early Stopping and Batching...")
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

    # Store losses for plotting later
    train_losses_all.append(avg_train_loss)
    val_losses_all.append(avg_val_loss)

    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "C:/Users/kdhas/Desktop/Research New/models/new models/new/tgru_traffic_model.pth")
        print(f"✅ Saved best model at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

# After training, plot the loss curves
plt.figure(figsize=(10,6))
plt.plot(train_losses_all, label='Train Loss')
plt.plot(val_losses_all, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

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

    # Calculate metrics per output
    mae = mean_absolute_error(targets_unscaled, preds_unscaled, multioutput='raw_values')
    mse = mean_squared_error(targets_unscaled, preds_unscaled, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_unscaled, preds_unscaled, multioutput='raw_values')

    print(f"\n📊 {label} Evaluation:")
    print(f"MAE - Packet Rate: {mae[0]:.4f}, Byte Rate: {mae[1]:.4f}")
    print(f"MSE - Packet Rate: {mse[0]:.4f}, Byte Rate: {mse[1]:.4f}")
    print(f"RMSE  - Packet Rate: {rmse[0]:.4f}, Byte Rate: {rmse[1]:.4f}")
    print(f"R²  - Packet Rate: {r2[0]:.4f}, Byte Rate: {r2[1]:.4f}")

    # Plot predictions vs actuals
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(targets_unscaled[:max_plot, 0], label='Actual Packet Rate')
    plt.plot(preds_unscaled[:max_plot, 0], label='Predicted Packet Rate')
    plt.title(f"{label} Set: Packet Rate")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(targets_unscaled[:max_plot, 1], label='Actual Byte Rate')
    plt.plot(preds_unscaled[:max_plot, 1], label='Predicted Byte Rate')
    plt.title(f"{label} Set: Byte Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ------------------------- Final Results -------------------------
evaluate_and_plot(X_train_tensor, y_train_tensor, delta_train, "Train")
evaluate_and_plot(X_val_tensor, y_val_tensor, delta_val, "Validation")
