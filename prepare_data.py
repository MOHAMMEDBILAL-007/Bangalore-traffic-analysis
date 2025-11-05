import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("D:\learning\Bangalore-traffic-analysis\Banglore_traffic_Dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])

holidays = pd.read_csv("D:\learning\Bangalore-traffic-analysis\holidays_2022_2026.csv")
holidays["Date"] = pd.to_datetime(holidays["Date"])

df = df.merge(holidays, on="Date", how="left")
df["IsHoliday"] = df["Holiday"].notna().astype(int)
df["Holiday"] = df["Holiday"].fillna("None")

df["DayOfWeek"] = df["Date"].dt.dayofweek  

cat_cols = ["Area Name", "Road/Intersection Name", "Weather Conditions",
            "Roadwork and Construction Activity", "Holiday"]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

df.to_csv("traffic_prepared.csv", index=False)
print(" traffic_prepared.csv created")
print(" encoders.pkl saved")
