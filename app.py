import time
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model

# --- 1. Initialization ---
app = Flask(__name__)
# Use a compatible async mode for Python 3.13 to avoid eventlet/gevent issues
socketio = SocketIO(app, async_mode='threading')

# --- 2. Load Model and Preprocessing Data ---
print("Loading model and preprocessing data...")
model = load_model("cnn_bilstm_ids.keras")
with open("feature_order.json", "r") as f:
    feature_order = json.load(f)
with open("minmax.json", "r") as f:
    minmax = json.load(f)
mins = pd.Series(minmax['min'])
maxs = pd.Series(minmax['max'])

# Define the original columns and categorical features
# NOTE: 'subclass' (label) is intentionally NOT included here
original_cols = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
categorical_features = ['protocol_type', 'service', 'flag']
class_names = ["DoS", "Normal", "Probe", "R2L", "U2R"]  # Matched to your training script

print("Model and data loaded successfully!")

# --- 3. Preprocessing Function ---
def preprocess_input(data_row):
    """Preprocesses a single row of network data for prediction."""
    # Build a 1-row DataFrame with the raw strings
    df_row = pd.DataFrame([data_row], columns=original_cols)

    # One-hot encode categorical features (protocol_type, service, flag)
    for col in categorical_features:
        dummies = pd.get_dummies(df_row[col], prefix=col, drop_first=False)
        df_row = pd.concat([df_row.drop(columns=[col]), dummies], axis=1)

    # Ensure all training-time columns exist, then enforce exact order
    for col in feature_order:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_order]

    # --- Robust numeric handling ---
    # Normalize ONLY columns that appear in minmax.json indices
    num_cols = [c for c in df_row.columns if c in mins.index]

    # Cast to numeric (coerce bad values to NaN), then fill NaN with mins
    df_row[num_cols] = df_row[num_cols].apply(pd.to_numeric, errors='coerce')
    df_row[num_cols] = df_row[num_cols].fillna(mins[num_cols])

    # Vectorized min-max scaling with aligned indices
    scale = (maxs[num_cols] - mins[num_cols]).replace(0, 1)  # avoid divide-by-zero
    df_row[num_cols] = (df_row[num_cols] - mins[num_cols]) / scale

    # Optional: clip to [0, 1]
    df_row[num_cols] = df_row[num_cols].clip(0.0, 1.0)

    # Reshape for the model
    processed_data = df_row.values.astype(np.float32)
    return np.reshape(processed_data, (1, processed_data.shape[1], 1))

# --- 4. Real-time Data Streaming ---
def data_stream():
    """Reads the test file and streams data to the client."""
    print("Starting data stream...")
    with open('KDDTest+.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # In NSL-KDD KDD*+.txt, the last two fields are: label, difficulty.
            # We drop both to keep only the 41 feature columns.
            raw_data_values = line.split(',')[:-2]

            # Preprocess the data for the model
            processed_data = preprocess_input(raw_data_values)

            # Get model prediction
            prediction_probs = model.predict(processed_data, verbose=0)
            prediction_index = np.argmax(prediction_probs, axis=1)[0]
            prediction_class = class_names[prediction_index]

            # Prepare data to send to the frontend
            data_to_send = {
                'raw_data': line,
                'prediction': prediction_class,
                'confidence': f"{prediction_probs[0][prediction_index] * 100:.2f}%"
            }
            socketio.emit('new_data', data_to_send)
            print(f"Sent: {prediction_class} -> {line}")
            time.sleep(0.5)  # Simulate real-time delay
    print("Data stream finished.")

# --- 5. Flask Routes and SocketIO Events ---
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Starts the data stream when a client connects."""
    print('Client connected!')
    # Use start_background_task to run the stream in a separate thread
    socketio.start_background_task(target=data_stream)

if __name__ == '__main__':
    socketio.run(app, debug=True)
