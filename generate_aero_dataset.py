# generate_mclaren_aero_dataset.py
import pandas as pd
import numpy as np

# Constants
rho_air = 1.225  # kg/m^3
mass_kg = 798    # approximate McLaren car mass
frontal_area_m2 = 1.2  # frontal area estimate
mu_tire = 1.7    # tire grip coefficient (soft/medium)
g = 9.81

# Load telemetry CSV
# Replace with your actual McLaren telemetry file
telemetry_file = "data/telemetry_NOR_Azerbaijan_2025.csv"
df = pd.read_csv(telemetry_file)

# Remove any rows with zero speed
df = df[df["Speed"] > 0].copy()

# Add some approximate setup values (for now, random within realistic range)
np.random.seed(42)
df["wing_angle"] = np.random.uniform(5, 25, size=len(df))   # degrees
df["ride_height"] = np.random.uniform(0.03, 0.07, size=len(df))  # meters
df["mode"] = np.random.choice(["X", "Z", "baseline"], size=len(df))

# Convert speed to m/s from km/h if needed
if df["Speed"].max() > 100:  # assume km/h
    df["speed_m_s"] = df["Speed"] * 1000 / 3600
else:
    df["speed_m_s"] = df["Speed"]

# Compute approximate Cd from straight sections
# P = 800 kW engine power estimate (for simplicity)
P_W = 800e3
df["Cd"] = (2 * P_W) / (rho_air * frontal_area_m2 * df["speed_m_s"]**3)
df["Cd"] = df["Cd"].clip(0, 1.5)  # sanity clip

# Compute approximate Cl from cornering speed
# Using v_corner = sqrt(mu*(m*g + L)*r/m) => L = v^2 * m / r - m*g*mu
# For simplicity, assume radius = 50m for all samples
radius = 50.0
df["Cl"] = ((df["speed_m_s"]**2) * mass_kg / radius - mass_kg * g * mu_tire) / (0.5 * rho_air * frontal_area_m2 * df["speed_m_s"]**2)
df["Cl"] = df["Cl"].clip(0, 5)  # sanity clip

# Select only necessary columns
dataset = df[["wing_angle", "ride_height", "mode", "Cd", "Cl"]]

# Save dataset for model training
dataset.to_csv("data/mclaren_aero_dataset.csv", index=False)
print("Saved McLaren aero dataset:", dataset.shape)
