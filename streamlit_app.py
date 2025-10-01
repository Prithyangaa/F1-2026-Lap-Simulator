"""
Streamlit UI for LapSimulator with 3D car & ML aero visualization.
Provides interactive controls, plotting, GA optimization, and force visualization.
"""
import json
import sqlite3
import random
from typing import Dict, Any
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from simulator import LapSimulator
import numpy as np


# -------------------------
# Database config
# -------------------------
DB_FILE = "runs.db"
DB_INIT_SQL = open("db_init.sql").read()

# -------------------------
# Default track
# -------------------------
DEFAULT_TRACK = {
    "name": "Monza-like",
    "lap_length_m": 5800.0,
    "straights_length_m": 1700.0,
    "corners": [
        {"radius_m": 120.0, "type": "fast"},
        {"radius_m": 60.0, "type": "medium"},
        {"radius_m": 40.0, "type": "slow"},
        {"radius_m": 80.0, "type": "fast"},
        {"radius_m": 50.0, "type": "medium"},
        {"radius_m": 30.0, "type": "slow"}
    ]
}

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Aero Mode Lap Simulator", layout="wide")
st.title("Aero-mode Lap Simulator — X/Z Modes (2026)")

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Car parameters")
wing_angle = st.sidebar.slider("Wing angle (deg)", 0.0, 30.0, 15.0, 0.5)
ride_height = st.sidebar.slider("Ride height (m)", 0.02, 0.12, 0.05, 0.005)
engine_power_kw = st.sidebar.number_input("Engine power (kW)", 100.0, 1200.0, 800.0, 10.0)
mass_kg = st.sidebar.number_input("Car mass (kg)", 600.0, 1000.0, 795.0, 1.0)
fuel_kg = st.sidebar.number_input("Fuel load (kg)", 0.0, 120.0, 5.0, 0.1)
tire_compound = st.sidebar.selectbox("Tire compound", ["soft", "medium", "hard"])
frontal_area = st.sidebar.number_input("Frontal area (m²)", 0.8, 2.0, 1.2, 0.01)

st.sidebar.header("Track & Environment")
track_choice = st.sidebar.selectbox("Select track", [DEFAULT_TRACK["name"], "Custom"])
if track_choice == DEFAULT_TRACK["name"]:
    track = DEFAULT_TRACK.copy()
    st.sidebar.write(f"Lap length (m): {track['lap_length_m']}")
    st.sidebar.write(f"Total straights (m): {track['straights_length_m']}")
else:
    track_json = st.sidebar.text_area("Paste track JSON", json.dumps(DEFAULT_TRACK, indent=2), height=200)
    try:
        track = json.loads(track_json)
    except Exception:
        st.sidebar.error("Invalid JSON, using default track.")
        track = DEFAULT_TRACK.copy()

st.sidebar.header("Environment & Mode")
air_density = st.sidebar.slider("Air density (kg/m³)", 1.0, 1.4, 1.225, 0.001)
mode = st.sidebar.selectbox("Aero mode", ["X", "Z", "baseline"])


# -------------------------
# Instantiate simulator
# -------------------------
sim = LapSimulator()

# -------------------------
# Car input dict
# -------------------------
car_params = {
    "wing_angle_deg": wing_angle,
    "ride_height_m": ride_height,
    "engine_power_kW": engine_power_kw,
    "mass_kg": mass_kg,
    "fuel_kg": fuel_kg,
    "tire_compound": tire_compound,
    "frontal_area_m2": frontal_area,
    "air_density_kg_m3": air_density
}

# -------------------------
# Buttons: Simulate / Save / GA
# -------------------------
col1, col2, col3 = st.columns(3)
if "last_outputs" not in st.session_state:
    with st.spinner("Simulating..."):
        outputs = sim.simulate_lap(car_params, track, mode=mode)
        st.session_state["last_outputs"] = outputs
        st.session_state["last_inputs"] = car_params

with col2:
    if st.button("Save run"):
        if "last_outputs" in st.session_state:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.executescript(DB_INIT_SQL)
            cur.execute(
                "INSERT INTO runs (mode, lap_time_sec, json_inputs, json_outputs) VALUES (?, ?, ?, ?)",
                (
                    mode,
                    float(st.session_state["last_outputs"]["lap_time_sec"]),
                    json.dumps(st.session_state["last_inputs"]),
                    json.dumps(st.session_state["last_outputs"])
                )
            )
            conn.commit()
            conn.close()
            st.success("Saved run to runs.db")
        else:
            st.warning("No simulation to save — run simulate first")


# -------------------------
# Initialize session state
# -------------------------
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = {}

if "last_outputs" not in st.session_state:
    st.session_state["last_outputs"] = {}

# -------------------------
# GA optimization (improved)
# -------------------------
def run_ga_optimize(simulator: LapSimulator, base_car: Dict[str, Any], track: Dict[str, Any], mode: str,
                    pop_size: int = 30, generations: int = 40):
    random.seed(42)

    # Search bounds (realistic)
    wa_min, wa_max = 0.0, 30.0        # wing angle in degrees
    rh_min, rh_max = 0.02, 0.12       # ride height in meters

    # Penalty thresholds
    MAX_DRAG = 50000      # N (50 kN upper bound)
    MAX_DOWNFORCE = 80000 # N (80 kN upper bound)
    MIN_DOWNFORCE = 500   # at least some downforce expected

    def random_individual():
        return {
            "wing_angle_deg": random.uniform(wa_min, wa_max),
            "ride_height_m": random.uniform(rh_min, rh_max)
        }

    def evaluate(ind):
        car = base_car.copy()
        car.update(ind)
        out = simulator.simulate_lap(car, track, mode=mode)
        lap_time = out["lap_time_sec"]

        # Add penalties for unphysical forces
        drag = max(out.get("drag_series", [0]))
        downforce = max(out.get("downforce_series", [0]))

        penalty = 0.0
        if drag > MAX_DRAG:
            penalty += (drag - MAX_DRAG) * 0.001
        if downforce > MAX_DOWNFORCE:
            penalty += (downforce - MAX_DOWNFORCE) * 0.001
        if downforce < MIN_DOWNFORCE:
            penalty += (MIN_DOWNFORCE - downforce) * 0.01

        # Ensure NaN/infinity doesn't break optimisation
        if not np.isfinite(lap_time):
            lap_time = 9999
            penalty += 1000

        return lap_time + penalty, out

    # Init population
    population = [random_individual() for _ in range(pop_size)]

    for gen in range(generations):
        scored = []
        for ind in population:
            lap_t, out = evaluate(ind)
            scored.append((lap_t, ind, out))
        scored.sort(key=lambda x: x[0])
        best = scored[0]

        st.write(f"Gen {gen+1}/{generations} — best lap {best[0]:.3f}s "
                 f"(wing={best[1]['wing_angle_deg']:.2f}, ride={best[1]['ride_height_m']:.3f})")

        # Selection
        keep = max(2, pop_size // 5)
        parents = [x[1] for x in scored[:keep]]
        new_pop = parents.copy()

        # Crossover + mutation
        while len(new_pop) < pop_size:
            a, b = random.sample(parents, 2)
            child = {
                "wing_angle_deg": (a["wing_angle_deg"] + b["wing_angle_deg"]) / 2.0,
                "ride_height_m": (a["ride_height_m"] + b["ride_height_m"]) / 2.0
            }
            if random.random() < 0.2:
                child["wing_angle_deg"] += random.uniform(-2.0, 2.0)
            if random.random() < 0.2:
                child["ride_height_m"] += random.uniform(-0.005, 0.005)

            # Clamp values within bounds
            child["wing_angle_deg"] = min(max(child["wing_angle_deg"], wa_min), wa_max)
            child["ride_height_m"] = min(max(child["ride_height_m"], rh_min), rh_max)
            new_pop.append(child)

        population = new_pop

        # Final evaluation
        final_scored = []
        for ind in population:
            lap_t, out = evaluate(ind)
            final_scored.append((lap_t, ind, out))
        final_scored.sort(key=lambda x: x[0])
        best = final_scored[0]
        best_lap, best_ind, best_out = scored[0]
        result = {
            "wing_angle_deg": best_ind["wing_angle_deg"],
            "ride_height_m": best_ind["ride_height_m"],
            "outputs": best_out,
            "lap_time_sec": best_lap
        }
        return result, final_scored

# Button trigger
with col3:
    if st.button("Optimize (GA)"):
        st.info("Running GA (pop=30, gens=40)...")
        best, history = run_ga_optimize(sim, car_params, track, mode, pop_size=30, generations=40)
        st.json(best)
        st.session_state["last_outputs"] = best["outputs"]
        st.session_state["last_inputs"].update({
            "wing_angle_deg": best["wing_angle_deg"],
            "ride_height_m": best["ride_height_m"]
        })
        st.success("GA finished")

# -------------------------
# Display results
# -------------------------
if "last_outputs" in st.session_state:
    out = st.session_state["last_outputs"]

    st.markdown("## Lap Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Lap time (s)", f"{out['lap_time_sec']:.3f}")
    col2.metric("Max speed (km/h)", f"{out['max_speed_kmh']:.1f}")
    col3.metric("Avg corner speed (km/h)", f"{out['avg_corner_speed_kmh']:.1f}")
    col4.metric("Fuel (kg/lap)", f"{out['fuel_used_kg_per_lap']:.2f}")
    col5.metric("Cd / Cl", f"{out['Cd']:.3f} / {out['Cl']:.3f}")

    # --- Graph at the bottom ---
    
    # --- Drag vs Downforce Graph ---
    df = pd.DataFrame({
        "Distance (m)": out["dist_series"],
        "Drag (N)": out["drag_series"],
        "Downforce (N)": out["downforce_series"]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Distance (m)"], y=df["Drag (N)"],
        mode="lines", name="Drag", line=dict(color="red")
    ))
    fig.add_trace(go.Scatter(
        x=df["Distance (m)"], y=df["Downforce (N)"],
        mode="lines", name="Downforce", line=dict(color="blue")
    ))

    fig.update_layout(
        title="Drag & Downforce Across Lap",
        xaxis_title="Distance (m)",
        yaxis_title="Force (N)",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )

    st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Sidebar notes
# -------------------------
st.sidebar.markdown("**Notes & assumptions**")
st.sidebar.write("- Physics-based lap model with ML-predicted aero coefficients.")
st.sidebar.write("- Corner lengths estimated from radii and type.")
st.sidebar.write("- Fuel estimate: energy / efficiency model.")
st.sidebar.write("- GA optimizes wing angle & ride height only.")

# -------------------------
# Ensure DB at startup
# -------------------------
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.executescript(DB_INIT_SQL)
    conn.commit()
    conn.close()

ensure_db()
