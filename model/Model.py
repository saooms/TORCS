import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Configuration
MODEL_PATH = 'torcs_ai_model.h5'
SCALER_PATH = 'torcs_scaler.save'

def load_data(csv_path):
    """Load numeric data from CSV, trying multiple parsing methods."""
    try:
        df = pd.read_csv(csv_path, comment='#')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        if not df.empty:
            return df.values

    except Exception as e:
        print(f"Failed to load {csv_path}: {e}\n")
        return None

def prepare_data(data):
    """Split into features and labels with validation"""
    if data is None or len(data) == 0:
        raise ValueError("Invalid data input for preparation")

    y = data[:, :3]  # First 5 columns as actions
    X = data[:, 3:]  # Remaining columns as features (sensors + state)

    # Clip actions to valid ranges as per TORCS requirements
    y[:, 0] = np.clip(y[:, 0], 0, 1)    # Acceleration [0,1]
    y[:, 1] = np.clip(y[:, 1], 0, 1)    # Brake [0,1]
    y[:, 2] = np.clip(y[:, 2], -1, 1)   # Steering [-1,1]

    return X, y

def build_model(input_dim):
    """Create optimized neural network architecture"""
    model = Sequential([
        Dense(32, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
            Dense(3, activation='tanh')
    ])

    model.compile(
        # optimizer=Adam(learning_rate=0.001),
        optimizer="adam",
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()]
    )
    return model

def train(csv_path):
    # Load and validate data
    raw_data = load_data(csv_path)
    if raw_data is None:
        print("Failed to load training data")
        return None, None

    try:
        X, y = prepare_data(raw_data)
        print(f"Data loaded successfully. X: {X.shape}, y: {y.shape}")
    except Exception as e:
        print(f"Data preparation failed: {str(e)}")
        return None, None

    # Train/validation split
    xTrain, X_val, yTrain, yVal = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Feature scaling
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    X_val = scaler.transform(X_val) 
    joblib.dump(scaler, SCALER_PATH)

    # Build and train model
    model = build_model(xTrain.shape[1])

    print("\nTraining model...")
    history = model.fit(
        xTrain, yTrain,
        validation_data=(X_val, yVal),
        epochs=500,
        batch_size=256,
        verbose=1,
        callbacks = [
            EarlyStopping(min_delta=1e-4, patience=15)
]
    )

    # Save as HDF5
    model.save(MODEL_PATH)
    print(f"\nModel saved in HDF5 format to {MODEL_PATH}")

    # Load and summarize the model
    model = load_model(MODEL_PATH)
    model.summary()

    # Sample prediction
    sampleIdx = np.random.randint(0, len(X_val))
    sample = X_val[sampleIdx].reshape(1, -1)
    prediction = model.predict(sample, verbose=0)[0]

    print("\nSample Validation Prediction:")
    print(f"Input shape: {sample.shape}")
    print(f"Predicted: Accel={prediction[0]:.3f}, Brake={prediction[1]:.3f}, Steer={prediction[2]:.3f}")
    print(f"Actual:    Accel={yVal[sampleIdx,0]:.3f}, Brake={yVal[sampleIdx,1]:.3f}, Steer={yVal[sampleIdx,2]:.3f}")

    return model, scaler

if __name__ == "__main__":
    train("train_data/combined_data.csv")
