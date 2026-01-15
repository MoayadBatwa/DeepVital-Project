import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Attention, GlobalAveragePooling1D, Concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Data Generation (Simulating PhysioNet Sepsis Dataset)
def load_real_data():
    try:
        df = pd.read_csv('clinical_data.csv')
        print("Loaded existing data.")
    except FileNotFoundError:
        print("Generating synthetic PhysioNet-style data...")
        # Create 1000 patients with 24 hours of vitals each
        n_patients = 1000
        data = []
        for pid in range(n_patients):
            is_sepsis = np.random.rand() > 0.8 # 20% sepsis prevalence
            
            # Base vitals (Normal distribution)
            base_hr = np.random.normal(80, 10)
            base_sbp = np.random.normal(120, 15)
            
            for h in range(24):
                # Apply deterioration trend if sepsis
                trend = (h/24) if is_sepsis else 0
                
                hr = base_hr + (trend * 30) + np.random.normal(0, 5)
                sbp = base_sbp - (trend * 20) + np.random.normal(0, 5)
                o2 = 98 - (trend * 10) + np.random.normal(0, 2)
                resp = 18 + (trend * 5) + np.random.normal(0, 2)
                
                # Label is 1 only in the last 6 hours of sepsis cases
                label = 1 if (is_sepsis and h > 18) else 0
                
                data.append([pid, hr, sbp, o2, resp, label])
        
        df = pd.DataFrame(data, columns=['Patient_ID', 'HR', 'SBP', 'O2Sat', 'Resp', 'Label'])
        df.to_csv('clinical_data.csv', index=False)
    return df

# 2. Preprocessing
def preprocess_data(df, time_steps=24):
    grouped = df.groupby('Patient_ID')
    X, y = [], []
    for _, group in grouped:
        if len(group) >= time_steps:
            # Extract last 24h vitals
            vitals = group[['HR', 'SBP', 'O2Sat', 'Resp']].values[-time_steps:]
            label = group['Label'].values[-1]
            X.append(vitals)
            y.append(label)
    return np.array(X), np.array(y)

# Load and Process
df = load_real_data()
X, y = preprocess_data(df)

# Scale features (Critical for Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 4)).reshape(-1, 24, 4)

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Model Architecture (Bi-LSTM + Attention)
inputs = Input(shape=(24, 4))
# Bi-LSTM to understand temporal context (past & future context)
lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
# Attention to focus on critical time steps
attention = Attention()([lstm_out, lstm_out])
# Merge and simplify
concat = Concatenate()([lstm_out, attention])
gap = GlobalAveragePooling1D()(concat)
x = Dense(32, activation='relu')(gap)
x = Dropout(0.3)(x) # Prevent overfitting
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train and Save
print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

model.save('deepvital_model.h5')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved successfully.")