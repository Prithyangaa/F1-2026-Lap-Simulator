"""
Pytest unit tests for simulator physics functions.
"""

import math
import pytest
from simulator import LapSimulator, DEFAULTS

def test_compute_aero_defaults():
    sim = LapSimulator()
    Cd, Cl = sim.compute_aero(15.0, 0.05, mode="X", frontal_area_m2=1.2)
    # Based on defaults: sanity checks
    assert Cd > 0
    assert Cl > 0
    # X-mode should increase Cl vs baseline
    Cd_b, Cl_b = sim.compute_aero(15.0, 0.05, mode="baseline", frontal_area_m2=1.2)
    assert Cl > Cl_b

def test_corner_speed_reasonable():
    sim = LapSimulator()
    # sample values
    radius = 50.0
    mass = 800.0
    Cd, Cl = sim.compute_aero(10.0, 0.05, mode="baseline", frontal_area_m2=1.2)
    mu = sim.const["tire_mu"]["soft"]
    v = sim.corner_speed(radius, Cl, Cd, 1.2, mu, mass)
    assert v > 0
    # Check not absurd (less than 120 m/s ~ 432 km/h)
    assert v < 120.0

def test_terminal_velocity_monotonic():
    sim = LapSimulator()
    Cd_low, _ = sim.compute_aero(0.0, 0.06, "Z", 1.2)
    Cd_high, _ = sim.compute_aero(30.0, 0.03, "X", 1.2)
    v_low = sim.terminal_velocity(800.0, Cd_low, 1.2)
    v_high = sim.terminal_velocity(800.0, Cd_high, 1.2)
    assert v_low > 0 and v_high > 0
    # Lower Cd (Z-mode) should give higher terminal speed
    assert v_low >= v_high
# file: tests/test_simulator.py
"""
Pytest unit tests for simulator physics functions.
"""

import math
import pytest
from simulator import LapSimulator, DEFAULTS

def test_compute_aero_defaults():
    sim = LapSimulator()
    Cd, Cl = sim.compute_aero(15.0, 0.05, mode="X", frontal_area_m2=1.2)
    # Based on defaults: sanity checks
    assert Cd > 0
    assert Cl > 0
    # X-mode should increase Cl vs baseline
    Cd_b, Cl_b = sim.compute_aero(15.0, 0.05, mode="baseline", frontal_area_m2=1.2)
    assert Cl > Cl_b

def test_corner_speed_reasonable():
    sim = LapSimulator()
    # sample values
    radius = 50.0
    mass = 800.0
    Cd, Cl = sim.compute_aero(10.0, 0.05, mode="baseline", frontal_area_m2=1.2)
    mu = sim.const["tire_mu"]["soft"]
    v = sim.corner_speed(radius, Cl, Cd, 1.2, mu, mass)
    assert v > 0
    # Check not absurd (less than 120 m/s ~ 432 km/h)
    assert v < 120.0

def test_terminal_velocity_monotonic():
    sim = LapSimulator()
    Cd_low, _ = sim.compute_aero(0.0, 0.06, "Z", 1.2)
    Cd_high, _ = sim.compute_aero(30.0, 0.03, "X", 1.2)
    v_low = sim.terminal_velocity(800.0, Cd_low, 1.2)
    v_high = sim.terminal_velocity(800.0, Cd_high, 1.2)
    assert v_low > 0 and v_high > 0
    # Lower Cd (Z-mode) should give higher terminal speed
    assert v_low >= v_high
