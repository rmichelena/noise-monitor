-- Esquema v2 — un solo tipo de fila: "evento".
--
-- kind:
--   traffic  vehículo detectado por el tracker, sin trigger de audio
--   noise    trigger de audio sin vehículo asociado
--   both     ruido y vehículo que coinciden; un solo clip con la ventana unión
--
-- start_ts/end_ts son epoch y definen la ventana del clip; peak_ts es el
-- instante del trigger cuando lo hubo.

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('traffic', 'noise', 'both')),
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    peak_ts REAL,
    timestamp TEXT NOT NULL,          -- ISO local del inicio (orden y filtros)
    db_max REAL,
    db_avg REAL,
    trigger_type TEXT,
    sound_class TEXT,
    sound_category TEXT,
    sound_confidence REAL,
    vehicle_class TEXT,
    direction TEXT,
    lane TEXT,
    speed_kmh REAL,
    detection_score REAL,
    track_id INTEGER,                 -- id del track en el tracker
    clip_path TEXT,
    audio_muted INTEGER DEFAULT 0,
    pinned INTEGER DEFAULT 0,
    pinned_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_vehicle ON events(vehicle_class);
CREATE INDEX IF NOT EXISTS idx_events_db ON events(db_max);
CREATE INDEX IF NOT EXISTS idx_events_pinned ON events(pinned);

-- Nivel de fondo por minuto: permite calibrar umbrales con datos reales.
CREATE TABLE IF NOT EXISTS audio_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    minute TEXT NOT NULL UNIQUE,      -- ISO 8601 local, 'YYYY-MM-DDTHH:MM'
    db_avg REAL NOT NULL,
    db_max REAL NOT NULL,
    db_p95 REAL,
    blocks INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_al_minute ON audio_levels(minute);
