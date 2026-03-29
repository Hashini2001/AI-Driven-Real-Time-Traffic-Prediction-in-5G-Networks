import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ------------------------- Configuration -------------------------
WINDOW_SIZE = 10
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ------------------------- Load Data -------------------------
X = np.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_X_train.npy")
y = np.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_y_train.npy")

scaler_y = joblib.load("C:/Users/kdhas/Desktop/Research New/models/preprocess data/test_common_scaler_y.pkl")

# ------------------------- Sequence Creation -------------------------
def create_sequences(X, y, window_size):
    X_seqs, y_seqs = [], []
    for i in range(len(X) - window_size):
        X_seqs.append(X[i:i + window_size])
        y_seqs.append(y[i + window_size])
    return np.array(X_seqs), np.array(y_seqs)

X_seq, y_seq = create_sequences(X, y, WINDOW_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False)

# ------------------------- Model Definition -------------------------
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim)
        )
    def forward(self, t, x):
        return self.net(x)

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc
    def forward(self, x, t):
        return odeint(self.odefunc, x, t, method='dopri5')

class SeqNeuralODEPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim = 2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.odeblock = ODEBlock(ODEFunc(hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, t):
        _, h = self.rnn(x.to(DEVICE))
        h = h.squeeze(0)
        out = self.odeblock(h, t.to(DEVICE))
        return self.fc(out[-1])

# ------------------------- Training -------------------------
input_dim = X.shape[1]
model = SeqNeuralODEPredictor(input_dim=input_dim, output_dim=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
t = torch.tensor([0., 1.])  # Not on GPU to avoid memory fragmentation
train_losses_all = []
val_losses_all = []

best_val_loss = float('inf')
patience_counter = 0

print("\n🚀 Training with Early Stopping and Batching...")
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch, t)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
            val_pred = model(X_val_batch, t)
            val_loss = loss_fn(val_pred, y_val_batch).item()
            val_losses.append(val_loss)

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)

    # Store losses for plotting later
    train_losses_all.append(avg_train_loss)
    val_losses_all.append(avg_val_loss)

    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "C:/Users/kdhas/Desktop/Research New/models/new models/new/new_neural_ode_traffic_model.pth")
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

# ------------------------- Load Best Model -------------------------
model.load_state_dict(torch.load("C:/Users/kdhas/Desktop/Research New/models/new models/new/new_neural_ode_traffic_model.pth"))
model.eval()

# ------------------------- Evaluation -------------------------
def evaluate_and_plot(X_tensor, y_tensor, label, max_plot=500):
    pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE):
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            batch_pred = model(X_batch, t).cpu().numpy()
            pred.append(batch_pred)
            y_true.append(y_batch.cpu().numpy())

    pred = np.vstack(pred)
    y_true = np.vstack(y_true)

    # Step 1: Inverse scale (from normalized log1p scale back to log1p scale)
    pred_unscaled = scaler_y.inverse_transform(pred)
    y_true_unscaled = scaler_y.inverse_transform(y_true)

    # Compute per-target metrics
    mae = mean_absolute_error(y_true_unscaled, pred_unscaled, multioutput='raw_values')
    mse = mean_squared_error(y_true_unscaled, pred_unscaled, multioutput='raw_values')
    r2 = r2_score(y_true_unscaled, pred_unscaled, multioutput='raw_values')

    print(f"\n📊 {label} Evaluation:")
    print(f"MAE - Packet Rate: {mae[0]:.4f}, Byte Rate: {mae[1]:.4f}")
    print(f"MSE - Packet Rate: {mse[0]:.4f}, Byte Rate: {mse[1]:.4f}")
    print(f"R²  - Packet Rate: {r2[0]:.4f}, Byte Rate: {r2[1]:.4f}")

    # Plot both targets
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(y_true_unscaled[:max_plot, 0], label='Actual Packet Rate')
    plt.plot(pred_unscaled[:max_plot, 0], label='Predicted Packet Rate')
    plt.title(f"{label} Set: Packet Rate")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(y_true_unscaled[:max_plot, 1], label='Actual Byte Rate')
    plt.plot(pred_unscaled[:max_plot, 1], label='Predicted Byte Rate')
    plt.title(f"{label} Set: Byte Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



# ------------------------- Final Results -------------------------
evaluate_and_plot(X_train_tensor, y_train_tensor, "Train")
evaluate_and_plot(X_val_tensor, y_val_tensor, "Validation")
