
# Aero-mode Lap Simulator (2026) — lightweight demo

This repository is a self-contained lap-time simulator that models the drag/downforce tradeoff for 2026-style **X-mode** (high downforce) and **Z-mode** (low-drag) aero states.

## What you get
- `simulator.py` — LapSimulator class implementing physics & lap integration.
- `streamlit_app.py` — Streamlit UI to run simulations, save runs to SQLite, and run a simple GA.
- `db_init.sql` — SQLite DDL to create `runs` table.
- `tests/test_simulator.py` — basic pytest tests for core physics functions.
- `requirements.txt` — python dependencies.

## Quick start
1. Create virtual environment and install:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
