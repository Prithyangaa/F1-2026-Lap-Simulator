CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT,
    lap_time_sec REAL,
    json_inputs TEXT,
    json_outputs TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);