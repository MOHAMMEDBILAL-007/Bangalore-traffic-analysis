import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import requests

CITY = "Bengaluru,IN"

model = tf.keras.models.load_model("traffic_lstm_model.keras")
scaler = pickle.load(open("scaler.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

df = pd.read_csv("traffic_prepared.csv")
holidays = pd.read_csv("holidays_2022_2026.csv")
holidays["Date"] = pd.to_datetime(holidays["Date"])

print("\nBengaluru Traffic Predictor — LIVE Weather + Holiday + LSTM\n")

date_str = input("Enter date (YYYY-MM-DD): ")
date = pd.to_datetime(date_str)

row = holidays[holidays["Date"] == date]
isHoliday = 1 if not row.empty else 0
holiday_name = row["Holiday"].values[0] if isHoliday else "None"
print("Holiday:", holiday_name)

dayOfWeek = date.dayofweek

areas = sorted(encoders["Area Name"].classes_)
for i,a in enumerate(areas,1): print(f"{i}. {a}")
area = areas[int(input("\nSelect Area: "))-1]

roads = df[df["Area Name"] == encoders["Area Name"].transform([area])[0]]["Road/Intersection Name"].unique()
roads = encoders["Road/Intersection Name"].inverse_transform(roads)
for i,r in enumerate(roads,1): print(f"{i}. {r}")
road = roads[int(input("\nSelect Road: "))-1]

speed = float(input("Enter current speed (km/h): "))
congestion = float(input("Enter congestion (0-100): "))

print("\n Fetching live weather from Open-Meteo...")

try:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "current": "weather_code"
    }

    response = requests.get(url, params=params).json()
    code = response["current"]["weather_code"]

    if code in [0, 1]:           weather = "Clear"
    elif code in [2, 3]:         weather = "Cloudy"
    elif code in [45, 48]:       weather = "Fog"
    elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]:
        weather = "Rainy"
    else:                        weather = "Clear"

    print(f" Weather Detected: {weather}")

except Exception:
    print(" Weather fetch failed — select manually:")
    ws = list(encoders["Weather Conditions"].classes_)
    for i, x in enumerate(ws,1): print(f"{i}. {x}")
    weather = ws[int(input("Select Weather: ")) - 1]

roadwork = input("Roadwork? (Yes/No): ").capitalize()

sample = {
 "Area Name": encoders["Area Name"].transform([area])[0],
 "Road/Intersection Name": encoders["Road/Intersection Name"].transform([road])[0],
 "Average Speed": speed,
 "Congestion Level": congestion,
 "Weather Conditions": encoders["Weather Conditions"].transform([weather])[0]
       if weather in encoders["Weather Conditions"].classes_ else 0,
 "Roadwork and Construction Activity":
       encoders["Roadwork and Construction Activity"].transform([roadwork])[0]
       if roadwork in encoders["Roadwork and Construction Activity"].classes_ else 0,
 "Holiday": encoders["Holiday"].transform([holiday_name])[0],
 "IsHoliday": isHoliday,
 "DayOfWeek": dayOfWeek,
}

for col in df.drop(columns=["Travel Time Index","Date"]).columns:
    if col not in sample: sample[col] = df[col].mean()

X = pd.DataFrame([sample])

X = X[df.drop(columns=["Travel Time Index","Date"]).columns]

X = scaler.transform(X).reshape(1, 1, -1)

pred = model.predict(X)[0][0]

road_len = 2.5
time_min = (road_len/speed)*60*pred

print("\n=========== RESULT =============")
print(f" {area} → {road}")
print(f" {date_str} | {holiday_name}")
print(f" Weather: {weather}")
print(f" TTI: {pred:.2f}")
print(f" Travel Time: {time_min:.1f} min")
print("================================")

if pred > 1.5: print(" Heavy Traffic — consider alternate routes!")
elif pred > 1.0: print(" Moderate Traffic")
else: print(" Smooth Traffic — Enjoy the ride!")
