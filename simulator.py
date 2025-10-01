"""
LapSimulator: F1-style lap-time simulator with physics approximations.
Now includes continuous lap trace building + plotting utilities.
"""

import math
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 9.81
DEFAULT_RHO = 1.225
FUEL_ENERGY_J_PER_KG = 43e6
ENGINE_THERMAL_EFFICIENCY = 0.30

DEFAULTS = {
    "Cd0": 0.90,
    "Cl0": 3.0,
    "k_d": 0.008,
    "k_l": 0.06,
    "ride_ref": 0.06,
    "h_d": 1.0,
    "h_l": 8.0,
    "modes": {
        "X": {"cd_factor": 1.08, "cl_factor": 1.12},
        "Z": {"cd_factor": 0.90, "cl_factor": 0.85},
        "baseline": {"cd_factor": 1.0, "cl_factor": 1.0}
    },
    "tire_mu": {"soft": 1.60, "medium": 1.45, "hard": 1.30},
    "dt": 0.1,
    "decel_max": 12.0,
    "accel_max": 12.0
}


@dataclass
class Corner:
    radius_m: float
    type: str


class LapSimulator:
    def __init__(self, constants: Dict[str, Any] = None):
        self.const = DEFAULTS.copy()
        if constants:
            self.const.update(constants)

    # -------------------------
    # Aerodynamics
    # -------------------------
    def compute_aero(self, wing_angle_deg, ride_height_m, mode="baseline", frontal_area_m2=1.2):
        Cd = (self.const["Cd0"] +
              self.const["k_d"] * wing_angle_deg +
              self.const["h_d"] * (self.const["ride_ref"] - ride_height_m))
        Cl = (self.const["Cl0"] +
              self.const["k_l"] * wing_angle_deg +
              self.const["h_l"] * (self.const["ride_ref"] - ride_height_m))
        mode = mode if mode in self.const["modes"] else "baseline"
        Cd *= self.const["modes"][mode]["cd_factor"]
        Cl *= self.const["modes"][mode]["cl_factor"]
        return Cd, Cl

    def drag_force(self, v, Cd, A, rho=DEFAULT_RHO):
        return 0.5 * rho * Cd * A * v * v

    def downforce(self, v, Cl, A, rho=DEFAULT_RHO):
        return 0.5 * rho * Cl * A * v * v

    # -------------------------
    # Cornering
    # -------------------------
    def corner_speed(self, radius_m, Cl, Cd, A, mu_tire, mass_kg, rho=DEFAULT_RHO):
        numerator = mu_tire * G * radius_m
        denom_term = mu_tire * 0.5 * rho * Cl * A * radius_m / mass_kg
        denom = max(1e-6, 1.0 - denom_term)
        v_sq = numerator / denom
        return math.sqrt(max(v_sq, 0.0))

    # -------------------------
    # Straight-line integration
    # -------------------------
    def straight_time(self, distance_m, v_start, v_target,
                      engine_power_kW, Cd, Cl, mass_kg, A, rho=DEFAULT_RHO, dt=None):
        if dt is None: dt = self.const["dt"]
        P = engine_power_kW * 1000.0
        decel_max, accel_max = self.const["decel_max"], self.const["accel_max"]

        s, v, t = 0.0, max(0.0, v_start), 0.0
        trace = [(0.0, v)]
        while s < distance_m:
            remain = distance_m - s
            if v > v_target:
                d_brake = (v * v - v_target * v_target) / (2.0 * decel_max)
            else:
                d_brake = 0.0

            if d_brake >= remain:
                a = -decel_max
            else:
                eff_v = max(v, 0.5)
                F_drive = P / eff_v
                F_drag = self.drag_force(v, Cd, A, rho)
                a = (F_drive - F_drag) / mass_kg
                a = np.clip(a, -decel_max, accel_max)

            v = max(0.0, v + a * dt)
            s += v * dt
            t += dt
            trace.append((s, v))

            if t > 500:  # safety cutoff
                break

        return t, v, max([vv for _, vv in trace]), trace

    # -------------------------
    # Lap simulation with trace
    # -------------------------
    def simulate_lap(self, car: Dict[str, Any], track: Dict[str, Any], mode="baseline"):
        wing, ride = car.get("wing_angle_deg", 15), car.get("ride_height_m", 0.05)
        power, mass, fuel = car.get("engine_power_kW", 800), car.get("mass_kg", 795), car.get("fuel_kg", 5)
        tire, A, rho = car.get("tire_compound", "soft"), car.get("frontal_area_m2", 1.2), car.get("air_density_kg_m3", DEFAULT_RHO)
        Cd, Cl = self.compute_aero(wing, ride, mode, A)
        mu = self.const["tire_mu"].get(tire, 1.45)

        corners = [Corner(float(c["radius_m"]), c.get("type", "medium")) for c in track.get("corners", [])]
        lap_len = track.get("lap_length_m", 5000.0)
        straights_total = track.get("straights_length_m", lap_len * 0.3)
        straight_each = straights_total / max(1, len(corners))

        v, total_time, max_speed = 0.0, 0.0, 0.0
        lap_trace = []
        distance_accum = 0.0
        corner_speeds = []

        # --- New series logs for plotting ---
        dist_series = []
        speed_series = []
        accel_series = []
        drag_series = []
        downforce_series = []

        # --- Simulation loop ---
        for i, corner in enumerate(corners):
            # Straight before this corner
            v_corner = self.corner_speed(corner.radius_m, Cl, Cd, A, mu, mass, rho)
            t_straight, v_end, v_max, trace = self.straight_time(
                straight_each, v, v_corner, power, Cd, Cl, mass, A, rho
            )
            total_time += t_straight
            max_speed = max(max_speed, v_max)

            # Log each point in the straight trace
            for s, vv in trace:
                distance = distance_accum + s
                accel = (power * 1000 / max(vv, 1e-6) - 0.5 * rho * Cd * A * vv**2) / mass
                drag = 0.5 * rho * Cd * A * vv**2
                downforce = 0.5 * rho * Cl * A * vv**2

                dist_series.append(distance)
                speed_series.append(vv * 3.6)   # m/s → km/h
                accel_series.append(accel)
                drag_series.append(drag)
                downforce_series.append(downforce)

                lap_trace.append((distance, vv))

            # Corner dynamics
            angle = {"slow": math.pi/2, "medium": math.pi/3, "fast": math.pi/4}.get(corner.type, math.pi/3)
            corner_dist = corner.radius_m * angle
            t_corner = corner_dist / max(v_corner, 1e-6)
            total_time += t_corner
            distance_accum += straight_each + corner_dist

            # Log corner endpoint
            dist_series.append(distance_accum)
            speed_series.append(v_corner * 3.6)
            accel_series.append(0.0)
            drag_series.append(0.5 * rho * Cd * A * v_corner**2)
            downforce_series.append(0.5 * rho * Cl * A * v_corner**2)

            lap_trace.append((distance_accum, v_corner))
            corner_speeds.append(v_corner)
            v = v_corner

        # Fuel estimate
        fuel_used = (power * 1000 * total_time) / (FUEL_ENERGY_J_PER_KG * ENGINE_THERMAL_EFFICIENCY)

        return {
            "lap_time_sec": total_time,
            "max_speed_kmh": max_speed * 3.6,
            "avg_corner_speed_kmh": np.mean(corner_speeds) * 3.6 if corner_speeds else 0,
            "fuel_used_kg_per_lap": fuel_used,
            "Cd": Cd,
            "Cl": Cl,
            "trace": lap_trace,
            "dist_series": dist_series,
            "speed_series": speed_series,
            "accel_series": accel_series,
            "drag_series": drag_series,
            "downforce_series": downforce_series,
            "corner_speeds": corner_speeds
        }

    # -------------------------
    # Plotting
    # -------------------------
    def plot_lap_trace(self, results: Dict[str, Any]):
        trace = np.array(results["trace"])
        if trace.size == 0: return None
        s, v = trace[:, 0], trace[:, 1] * 3.6
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(s, v, lw=2, c="orange")
        ax.set_title("Speed Trace Along Lap")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Speed (km/h)")
        ax.grid(True, alpha=0.3)
        return fig

    def plot_cd_cl_tradeoff(self, Cd_vals, Cl_vals):
        fig, ax = plt.subplots()
        ax.plot(Cd_vals, Cl_vals, "o-", c="purple")
        ax.set_xlabel("Drag Coefficient (Cd)")
        ax.set_ylabel("Downforce Coefficient (Cl)")
        ax.set_title("Cd vs Cl Tradeoff")
        ax.grid(True, alpha=0.3)
        return fig


# -------------------------
# Manual test
# -------------------------
if __name__ == "__main__":
    sim = LapSimulator()
    car = {"wing_angle_deg": 15, "ride_height_m": 0.05, "engine_power_kW": 800, "mass_kg": 795,
           "fuel_kg": 5, "tire_compound": "soft", "frontal_area_m2": 1.2}
    track = {"lap_length_m": 5800, "straights_length_m": 1700,
             "corners": [{"radius_m": 120, "type": "fast"}, {"radius_m": 60, "type": "medium"}, {"radius_m": 40, "type": "slow"}]}
    out = sim.simulate_lap(car, track, mode="X")
    print(json.dumps(out, indent=2))
    sim.plot_lap_trace(out)
    plt.show()
