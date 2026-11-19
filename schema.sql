-- Schema para el sistema de monitoreo de ruido urbano v2
-- Ejecutar: sqlite3 data/events.db < schema.sql
--
-- Dos tablas principales:
--   vehicle_passages: registro continuo de cada vehículo que cruza el cuadro (tracker)
--   noise_events: eventos de ruido con clip (vehicular y no vehicular)

-- Registro continuo de vehículos detectados por el tracker
CREATE TABLE IF NOT EXISTS vehicle_passages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601 del momento en que el vehículo entró al cuadro
    timestamp_exit TEXT,               -- ISO 8601 cuando salió del cuadro
    vehicle_class TEXT NOT NULL,       -- 'car', 'truck', 'bus', 'motorcycle', 'bicycle'
    lane TEXT,                         -- mitad de imagen: 'left', 'right'
    direction TEXT,                    -- sentido real: 'ltr' o 'rtl'
    speed_kmh REAL,                    -- velocidad media estimada por píxeles
    detection_score REAL,              -- Score promedio del modelo durante el tracking
    db_at_time REAL,                   -- dB(A) al momento de la detección
    clip_path TEXT,                    -- clip del pasaje (vehículos en movimiento)
    pinned INTEGER DEFAULT 0,          -- 1 si el dueño lo marcó
    pinned_at TEXT,
    noise_event_id INTEGER,            -- FK a noise_events si este vehículo generó un evento ruidoso
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (noise_event_id) REFERENCES noise_events(id)
);

-- Eventos de ruido (vehiculares y no vehiculares) con clip de audio/video
CREATE TABLE IF NOT EXISTS noise_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601
    db_max REAL NOT NULL,              -- Nivel máximo ponderado A
    db_avg REAL,                       -- Nivel promedio en ventana
    trigger_type TEXT,                 -- 'peak', 'sustained'
    sound_class TEXT,                  -- Clase YAMNet raw: 'Vehicle', 'Dog', 'Jackhammer', etc.
    sound_category TEXT,               -- Categoría agrupada: 'vehicle','construction','animal','alarm','human','other'
    sound_confidence REAL,             -- Score de YAMNet (0.0-1.0)
    vehicle_class TEXT,                -- Solo si vehicular: 'motorcycle', 'car', 'truck', 'bus', 'unknown'
    lane TEXT,                         -- 'left', 'right', 'both', 'none'
    confidence TEXT,                   -- 'high', 'medium', 'low' (confianza en atribución visual)
    passage_id INTEGER,                -- FK a vehicle_passages si se correlacionó con un vehículo
    clip_path TEXT,                    -- Ruta relativa al clip
    audio_muted INTEGER DEFAULT 0,      -- 1 si se eliminó el audio (voz)
    pinned INTEGER DEFAULT 0,           -- 1 si el dueño lo marcó como interesante
    pinned_at TEXT,
    duration_ms INTEGER,               -- Duración del clip en ms
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (passage_id) REFERENCES vehicle_passages(id)
);

-- Migración para bases existentes (falla en silencio si ya existe)
ALTER TABLE vehicle_passages ADD COLUMN direction TEXT;
ALTER TABLE vehicle_passages ADD COLUMN speed_kmh REAL;
ALTER TABLE vehicle_passages ADD COLUMN clip_path TEXT;
ALTER TABLE vehicle_passages ADD COLUMN pinned INTEGER DEFAULT 0;
ALTER TABLE vehicle_passages ADD COLUMN pinned_at TEXT;
ALTER TABLE noise_events ADD COLUMN audio_muted INTEGER DEFAULT 0;
ALTER TABLE noise_events ADD COLUMN pinned INTEGER DEFAULT 0;
ALTER TABLE noise_events ADD COLUMN pinned_at TEXT;

CREATE INDEX IF NOT EXISTS idx_vp_timestamp ON vehicle_passages(timestamp);
CREATE INDEX IF NOT EXISTS idx_vp_class ON vehicle_passages(vehicle_class);
CREATE INDEX IF NOT EXISTS idx_ne_timestamp ON noise_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_ne_sound_category ON noise_events(sound_category);
CREATE INDEX IF NOT EXISTS idx_ne_vehicle_class ON noise_events(vehicle_class);
CREATE INDEX IF NOT EXISTS idx_ne_db ON noise_events(db_max);

-- Migrar datos de la tabla vieja si existe
INSERT OR IGNORE INTO noise_events (id, timestamp, db_max, db_avg, trigger_type, vehicle_class, lane, confidence, clip_path, duration_ms)
    SELECT id, timestamp, db_max, db_avg, trigger_type, vehicle_class, lane, confidence, clip_path, duration_ms
    FROM events WHERE EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='events');

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
