import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 10000

# 1. Core Identifiers & Real-World Carrier Roster
shipment_ids = [f"SHP-{100000 + i}" for i in range(n_rows)]
carriers = np.random.choice(
    [
        "FedEx Express",
        "Canada Post",
        "CN Rail",
        "Challenger Motor Freight",
    ],
    size=n_rows,
    p=[0.28, 0.24, 0.24, 0.24],
)
origin_hubs = np.random.choice(
    [
        "Toronto_Hub",
        "Mississauga_Hub",
        "Montreal_Hub",
        "Vancouver_Hub",
        "Chicago_CrossBorder_Hub",
    ],
    size=n_rows,
    p=[0.30, 0.25, 0.15, 0.15, 0.15],
)
shipping_modes = np.random.choice(
    ["Ground", "Express", "Intermodal Rail", "LTL Freight"],
    size=n_rows,
    p=[0.45, 0.25, 0.15, 0.15],
)

# 2. Operational & Environmental Features
distance_km = np.random.randint(150, 3500, size=n_rows)
package_weight_kg = np.round(np.random.exponential(scale=18.0, size=n_rows) + 1.0, 2)
weather_severity = np.random.choice(
    ["Clear", "Rain/Snow", "Severe Winter Storm"], size=n_rows, p=[0.75, 0.20, 0.05]
)
is_peak_season = np.random.choice([0, 1], size=n_rows, p=[0.70, 0.30])

# Calibrated Warehouse Dwell Time (Hours stuck at origin loading dock)
base_dwell = np.where(
    carriers == "FedEx Express",
    np.random.normal(loc=8.0, scale=3.0, size=n_rows),
    np.random.normal(loc=14.5, scale=4.0, size=n_rows),
)
warehouse_dwell_hrs = np.clip(np.round(base_dwell, 1), 1.0, 48.0)

# 3. Calibrated SLA Delay Probability (~63.1% OTD for FedEx, ~40% OTD for others)
risk_score = (
    -2.25
    + (warehouse_dwell_hrs * 0.13)
    + np.where(carriers == "FedEx Express", -0.55, -0.05)
    + np.where(weather_severity == "Severe Winter Storm", 1.4, 0.0)
    + np.where(weather_severity == "Rain/Snow", 0.35, 0.0)
    + (is_peak_season * 0.50)
    + (distance_km / 2800.0)
)

delay_probability = 1 / (1 + np.exp(-risk_score))
sla_delayed = np.random.binomial(1, delay_probability)

# 4. Assemble & Save DataFrame
df = pd.DataFrame({
    "shipment_id": shipment_ids,
    "carrier": carriers,
    "origin_hub": origin_hubs,
    "shipping_mode": shipping_modes,
    "distance_km": distance_km,
    "package_weight_kg": package_weight_kg,
    "warehouse_dwell_hrs": warehouse_dwell_hrs,
    "weather_severity": weather_severity,
    "is_peak_season": is_peak_season,
    "sla_delayed": sla_delayed,
})

df.to_csv("supply_chain_shipments.csv", index=False)

# Quick Verification Check
print("Dataset created successfully! Shape:", df.shape)
print("\nOn-Time Delivery Rate by Carrier:")
print((1 - df.groupby("carrier")["sla_delayed"].mean()).round(3) * 100)
