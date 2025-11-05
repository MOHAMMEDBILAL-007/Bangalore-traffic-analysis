import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("traffic_prepared.csv")

target = "Travel Time Index"
X = df.drop(columns=[target, "Date"])
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)

X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1, X.shape[1])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

loss, mae = model.evaluate(X_test, y_test)
print(f" Loss:{loss:.4f} | MAE:{mae:.4f}")

y_pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred)
print(f" R²:{r2:.3f}")

model.save("traffic_lstm_model.keras")
print(" Model saved as traffic_lstm_model.keras")

plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss vs Val Loss"); plt.legend()

plt.subplot(1,3,2)
plt.plot(history.history["mae"])
plt.title("MAE")

plt.subplot(1,3,3)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title(f"Actual vs Predicted\nR²={r2:.2f}")

plt.tight_layout()
plt.show()
