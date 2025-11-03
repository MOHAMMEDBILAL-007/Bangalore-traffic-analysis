import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
 
model = tf.keras.models.load_model("traffic_model.keras")

df = pd.read_csv("processed_traffic_data.csv")

raw_df = pd.read_csv("Banglore_traffic_Dataset.csv")
categorical_cols = ["Area Name", "Road/Intersection Name", "Weather Conditions", "Roadwork and Construction Activity"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(raw_df[col].unique())
    encoders[col] = le

scaler = StandardScaler()
X = df.drop(columns=["Travel Time Index", "Date"])
scaler.fit(X)

area_roads = {
    "Indiranagar": ["100 Feet Road", "CMH Road"],
    "Whitefield": ["Marathahalli Bridge", "ITPL Main Road"],
    "Koramangala": ["Sony World Junction", "Sarjapur Road"],
    "M.G. Road": ["Trinity Circle", "Anil Kumble Circle"],
    "Jayanagar": ["Jayanagar 4th Block", "South End Circle"],
    "Hebbal": ["Hebbal Flyover", "Ballari Road"],
    "Yeshwanthpur": ["Yeshwanthpur Circle", "Tumkur Road"],
    "Electronic City": ["Silk Board Junction", "Hosur Road"]
}

print("\n Bengaluru Traffic Predictor \n")

areas = list(area_roads.keys())
for i, a in enumerate(areas, 1):
    print(f"{i}. {a}")

area_choice = int(input("\nSelect Area Number: "))
selected_area = areas[area_choice - 1]

print(f"\nAvailable Roads in {selected_area}:")
roads = area_roads[selected_area]
for i, r in enumerate(roads, 1):
    print(f"{i}. {r}")

road_choice = int(input("\nSelect Road Number: "))
selected_road = roads[road_choice - 1]

speed = float(input("\nEnter current average speed (km/h): "))
congestion = float(input("Enter congestion level (0–100): "))

weather_options = ["Clear", "Rainy", "Cloudy"]
for i, w in enumerate(weather_options, 1):
    print(f"{i}. {w}")
weather_choice = int(input("\nSelect Weather Condition: "))
weather = weather_options[weather_choice - 1]

roadwork = input("Any roadwork or construction activity? (Yes/No): ").capitalize()

sample = pd.DataFrame([{
    "Area Name": encoders["Area Name"].transform([selected_area])[0],
    "Road/Intersection Name": encoders["Road/Intersection Name"].transform([selected_road])[0],
    "Average Speed": speed,
    "Congestion Level": congestion,
    "Weather Conditions": encoders["Weather Conditions"].transform([weather])[0] if weather in encoders["Weather Conditions"].classes_ else 0,
    "Roadwork and Construction Activity": encoders["Roadwork and Construction Activity"].transform([roadwork])[0] if roadwork in encoders["Roadwork and Construction Activity"].classes_ else 0,
}])

#fill mising colums to match model input
for col in X.columns:
    if col not in sample.columns:
        sample[col] = df[col].mean()

sample = sample[X.columns]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0][0]


road_length_km = 2.5
time_minutes = (road_length_km / speed) * 60 * prediction 

print("\n============================")
print(f"Location: {selected_area} - {selected_road}")
print(f"Weather: {weather} | Roadwork: {roadwork}")
print(f"Predicted Travel Time Index: {prediction:.2f}")
print(f"Estimated Time to Pass Road: {time_minutes:.1f} minutes")
print("============================\n")

if prediction > 1.5:
    print(" Heavy Traffic Expected. Plan alternate routes.")
elif prediction > 1.0:
    print(" Moderate Traffic. Expect slight delays.")
else:
    print(" Smooth Flow. Roads are clear!")
