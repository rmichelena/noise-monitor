-- Schema para el sistema de monitoreo de ruido urbano v3
-- Ejecutar: sqlite3 data/events.db < schema.sql
--
-- Una sola tabla de eventos, un clip por fila:
--   kind = 'traffic'  pasaje del tracker sin disparo acústico
--   kind = 'noise'    disparo acústico sin pasaje correlacionado
--   kind = 'both'     disparo acústico correlacionado con un pasaje
--
-- Las bases con noise_events/vehicle_passages se migran con
-- scripts/migrate_unified.py, que las deja como *_legacy. init_database se
-- niega a arrancar sobre una base sin migrar.
--
-- Este DDL debe coincidir con EVENTS_DDL de scripts/migrate_unified.py
-- (tests/test_schema_cutover.py lo verifica).
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('traffic', 'noise', 'both')),
    timestamp TEXT NOT NULL,          -- ISO 8601 local, orden y filtros
    start_ts REAL NOT NULL,           -- epoch: ventana del evento
    end_ts REAL NOT NULL,
    peak_ts REAL,                     -- epoch del trigger de audio, si lo hubo
    db_max REAL,
    db_avg REAL,
    trigger_type TEXT,
    sound_class TEXT,
    sound_category TEXT,
    sound_confidence REAL,
    sound_labels_json TEXT,
    vehicle_class TEXT,
    direction TEXT,
    lane TEXT,
    speed_kmh REAL,
    detection_score REAL,
    track_id INTEGER,
    confidence TEXT,
    audio_muted INTEGER DEFAULT 0,
    duration_ms INTEGER,
    clip_path TEXT,
    pinned INTEGER DEFAULT 0,
    pinned_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_vehicle ON events(vehicle_class);
CREATE INDEX IF NOT EXISTS idx_events_db ON events(db_max);
CREATE INDEX IF NOT EXISTS idx_events_pinned ON events(pinned);
CREATE INDEX IF NOT EXISTS idx_events_sound_category ON events(sound_category);

-- Nivel de fondo por minuto: permite calibrar umbrales con datos reales
-- sin depender de clips (que solo existen cuando ya se disparó un evento).
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

-- Pasajes del tracker por minuto, clase y sentido. Se cuentan todos, se
-- graben o no como evento: los flags de `triggers` deciden qué filas y clips
-- se guardan en `events`, no qué se mide. `direction` = '' si no se supo.
CREATE TABLE IF NOT EXISTS traffic_counts (
    minute TEXT NOT NULL,             -- ISO 8601 local, 'YYYY-MM-DDTHH:MM' (entrada)
    vehicle_class TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT '',
    count INTEGER NOT NULL DEFAULT 0,
    speed_n INTEGER NOT NULL DEFAULT 0,
    speed_sum REAL NOT NULL DEFAULT 0,
    speed_max REAL,
    PRIMARY KEY (minute, vehicle_class, direction)
);

CREATE TABLE IF NOT EXISTS acoustic_trigger_decisions (
    event_table TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    decision_json TEXT NOT NULL,
    PRIMARY KEY (event_table, event_id)
);
