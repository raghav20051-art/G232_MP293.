
import pandas as pd

df = pd.read_csv("data/train.csv")
df["date"] = pd.to_datetime(df["date"])

# Check 1: Stuck sensor
print("\n========== STUCK SENSOR INVESTIGATION ==========")

period = df[
    (df["date"] >= "2026-04-01") &
    (df["date"] <= "2026-04-30")
]

print("\nDaily days_to_salary readings:")
print(
    period.groupby("date")["days_to_salary"]
    .agg(["min", "max", "nunique", "count"])
    .to_string()
)

print("\nDates where every row has the same reading:")
daily = df.groupby("date")["days_to_salary"].agg(
    ["min", "max", "nunique", "count"]
)
print(daily[daily["nunique"] == 1].to_string())

# Check 2: Target typos
print("\n========== TARGET TYPO INVESTIGATION ==========")

target = df["withdrawal_thousand_rs"]

sorted_target = target.sort_values().reset_index(drop=True)
gaps = sorted_target.diff()

print("\nLargest gaps between consecutive sorted target values:")
print(
    gaps.nlargest(20).to_string()
)

print("\nRecords with largest target values:")
print(
    df.nlargest(30, "withdrawal_thousand_rs")[
        ["date", "area", "atm_type", "withdrawal_thousand_rs"]
    ].to_string(index=False)
)

print("\nTarget statistics by area:")
print(
    df.groupby("area")["withdrawal_thousand_rs"]
    .describe()
)

print("\n========== TARGET GAP DETAILS ==========")

ordered = df.sort_values("withdrawal_thousand_rs").reset_index(drop=True)

ordered["gap"] = ordered["withdrawal_thousand_rs"].diff()

top_gaps = ordered.nlargest(15, "gap")

for index, row in top_gaps.iterrows():
    previous = ordered.iloc[index - 1]

    print(
        f"\nGap: {row['gap']:.2f}"
        f"\nPrevious value: {previous['withdrawal_thousand_rs']}"
        f"\nCurrent value: {row['withdrawal_thousand_rs']}"
        f"\nCurrent record: {row['record_id']}"
        f"\nCurrent date: {row['date']}"
        f"\nCurrent area: {row['area']}"
    )

print("\n========== LATER DATA TARGET COMPARISON ==========")

later = pd.read_csv("data/later.csv")

print("\nTrain target statistics:")
print(df["withdrawal_thousand_rs"].describe())

print("\nLater target statistics:")
print(later["withdrawal_thousand_rs"].describe())

print("\nSmallest train target values:")
print(df.nsmallest(20, "withdrawal_thousand_rs")[
    ["record_id", "area", "withdrawal_thousand_rs"]
].to_string(index=False))

print("\n========== TARGET BY AREA AND ATM TYPE ==========")

print(
    df.groupby(["area", "atm_type"])["withdrawal_thousand_rs"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
    .to_string()
)

print("\n========== TARGET VALUE DISTRIBUTION ==========")

print(
    pd.cut(
        df["withdrawal_thousand_rs"],
        bins=[0, 100, 200, 300, 400, 500, 600,
              700, 800, 900, 1000, 1100, 1200, 1300]
    ).value_counts().sort_index().to_string()
)

print("\n========== TARGET VALUES ABOVE 1000 ==========")

print(
    df[df["withdrawal_thousand_rs"] > 1000][
        ["record_id", "area", "atm_type", "withdrawal_thousand_rs"]
    ].to_string(index=False)
)

print("\n========== TARGET TYPO CANDIDATE CHECK ==========")

# Look for values that are approximately 10 times
# another plausible target value.
targets = df["withdrawal_thousand_rs"].sort_values().reset_index(drop=True)

candidates = []

for i, value in enumerate(targets):
    if value >= 1000:
        possible_original = value / 10

        nearest = (targets - possible_original).abs().idxmin()
        nearest_value = targets.iloc[nearest]

        if abs(nearest_value - possible_original) <= 10:
            candidates.append((value, possible_original, nearest_value))

print("Possible 10x target matches:")
for item in candidates:
    print(item)

print("\nNumber of possible matches:", len(candidates))

print("\n========== LOW TARGET RECORD INVESTIGATION ==========")

print(
    df[df["withdrawal_thousand_rs"] < 350][
        [
            "record_id",
            "date",
            "nearby_shops",
            "days_to_salary",
            "footfall_index",
            "distance_to_market_km",
            "avg_ticket_rs",
            "other_atms_1km",
            "atm_type",
            "area",
            "withdrawal_thousand_rs"
        ]
    ].sort_values("withdrawal_thousand_rs").to_string(index=False)
)
