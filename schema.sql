-- Schema para el sistema de monitoreo de ruido vehicular
-- Ejecutar: sqlite3 data/events.db < schema.sql

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601: 2026-02-08T14:30:45
    db_max REAL NOT NULL,             -- Nivel máximo ponderado A
    db_avg REAL,                      -- Nivel promedio en ventana
    vehicle_class TEXT,               -- 'motorcycle', 'car', 'truck', 'bus', 'unknown'
    lane TEXT,                        -- 'left', 'right', 'both', 'none'
    confidence TEXT,                  -- 'high', 'medium', 'low'
    speed_estimate REAL,             -- km/h estimados (nullable)
    clip_path TEXT,                   -- Ruta relativa al video
    duration_ms INTEGER,              -- Duración del clip en ms
    trigger_type TEXT,                -- 'peak', 'sustained', 'correlation'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_vehicle ON events(vehicle_class);
CREATE INDEX IF NOT EXISTS idx_db ON events(db_max);
