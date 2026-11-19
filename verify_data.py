
import pandas as pd

train = pd.read_csv("data/train.csv")
later = pd.read_csv("data/later.csv")
holdout = pd.read_csv("data/holdout_features.csv")

print("\n========== BASIC INFORMATION ==========")
print("Train shape:", train.shape)
print("Later shape:", later.shape)
print("Holdout shape:", holdout.shape)

print("\n========== 1. LEAKING COLUMN CHECK ==========")
print("Train columns:", train.columns.tolist())
print("Holdout columns:", holdout.columns.tolist())
print("Columns missing from holdout:",
      set(train.columns) - set(holdout.columns))

print("\n========== 2. UNIT MIX-UP CHECK ==========")
print(train.groupby("area")["distance_to_market_km"].describe())

print("\n========== 3. SENTINEL VALUE CHECK ==========")
print("avg_ticket_rs = -999 count:",
      (train["avg_ticket_rs"] == -999).sum())

print("\n========== 4. DUPLICATE ROW CHECK ==========")
print("Duplicate rows:", train.duplicated().sum())

print("\n========== 5. STUCK SENSOR CHECK ==========")
sensor = train[["date", "days_to_salary"]].copy()
sensor["date"] = pd.to_datetime(sensor["date"])

print(
    sensor.groupby("days_to_salary")["date"]
    .agg(["min", "max", "count"])
    .sort_values("count", ascending=False)
    .head(10)
)

print("\n========== 6. TARGET TYPO CHECK ==========")
target = train["withdrawal_thousand_rs"]
print(target.describe(percentiles=[0.90, 0.95, 0.99]))
print("Largest target values:")
print(target.nlargest(20).to_list())

print("\n========== 7. SPELLING VARIANTS CHECK ==========")
print("\nTrain ATM types:")
print(train["atm_type"].value_counts(dropna=False))

print("\nLater ATM types:")
print(later["atm_type"].value_counts(dropna=False))

print("\n========== VERIFICATION COMPLETE ==========")
