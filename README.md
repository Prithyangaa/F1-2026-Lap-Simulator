# Aero-Mode Lap Simulator (2026) — lightweight demo

This repository is a self-contained lap-time simulator that models the drag/downforce tradeoff for 2026-style **X-mode** (high downforce) and **Z-mode** (low-drag) aero states. It includes a physics-based lap model, ML-predicted aero coefficients, and a simple GA optimizer for wing angle and ride height.

## What you get

* `simulator.py` — LapSimulator class implementing physics & lap integration.
* `streamlit_app.py` — Streamlit UI to run simulations, auto-run on page load, save runs to SQLite, and run a simple GA.
* `db_init.sql` — SQLite DDL to create `runs` table.
* `tests/test_simulator.py` — basic pytest tests for core physics functions.
* `requirements.txt` — Python dependencies.

## Quick start

Clone the repository:

```bash
git clone https://github.com/<your-username>/f1-2026-lap-simulator.git
cd f1-2026-lap-simulator
```

Create virtual environment and install:

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

* The simulation **runs automatically on page load**.
* Adjust car parameters, track, or aero mode, then click **“Simulate Lap”** to rerun.
* GA optimization can be triggered from the interface to find optimal wing angle and ride height.

## Future plans

* Expand GA optimization to include **engine power and tire compound**.
* Add **more tracks and corner modeling** for better accuracy.
* Implement **real-time lap visualization** with speed, acceleration, drag, and downforce series plotted dynamically.
* Integrate **ML-enhanced predictions** for tire degradation and fuel efficiency.
* Package the simulator for **web deployment** with user accounts and saved simulations.
