
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = "D:\learning\Deep learning\Banglore_traffic_Dataset.csv"   # change path if needed
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("\nSample Data:\n", df.head())

print("\nMissing values:\n", df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)

categorical_cols = ["Area Name", "Road/Intersection Name", "Weather Conditions", "Roadwork and Construction Activity"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nEncoding complete!")

processed_file = "processed_traffic_data.csv"
df.to_csv(processed_file, index=False)
print(f"\nProcessed data saved as {processed_file}")
