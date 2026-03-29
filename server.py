# t-lstm-server/inference_server.py

import socket
import torch
import joblib
import numpy as np
from new_lstm import TLSTMPredictor

# Load model and preprocessing tools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TLSTMPredictor(input_dim=50)  
model.load_state_dict(torch.load("tlstm_traffic_model.pth", map_location=device))
model.to(device).eval()

scaler_y = joblib.load("test_common_scaler_y.pkl")
feature_columns = joblib.load("test_common_feature_columns.pkl")
time_delta_index = feature_columns.index("time_delta")

# Setup socket server
host = '0.0.0.0'
port = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host, port))
sock.listen(1)
print("🟢 T-LSTM Prediction Server Running on port 5005")

def predict_packet_byte_rate(x_seq_np):
    delta_seq_np = x_seq_np[:, :, time_delta_index:time_delta_index+1]
    x_tensor = torch.tensor(x_seq_np, dtype=torch.float32).to(device)
    delta_tensor = torch.tensor(delta_seq_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(x_tensor, delta_tensor)
    preds = output.cpu().numpy()
    preds_unscaled = scaler_y.inverse_transform(preds)
    preds_final = np.expm1(preds_unscaled)
    return preds_final[0]  # [packet_rate, byte_rate]
try:
    while True:
        conn, addr = sock.accept()
        print(f"🔌 Connected by {addr}")
        with conn:
            expected_bytes = 10*50*4
            data = b''
            while len(data) < expected_bytes:
                packet = conn.recv(expected_bytes - len(data))
                if not packet:
                    print("Incomplete or closed connection...")
                    break
                data += packet
                if len(data) != expected_bytes:
                    print(f"Expected {expected_bytes} bytes, got {len(data)}.")
                    continue
                try:
                    input_array = np.frombuffer(data, dtype=np.float32).reshape(1,10,-1)
                    prediction = predict_packet_byte_rate(input_array)
                    conn.sendall(prediction.astype(np.float32).tobytes())
                except Exception as e:
                    print(f"Prediction failed: {e}")
except KeyboardInterrupt:
    print("🛑 Server shutting down.")
    sock.close()