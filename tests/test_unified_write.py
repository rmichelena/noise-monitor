#!/usr/bin/env python3
"""
Escritura en la tabla única `events` (Slice B): pasaje → traffic, ruido →
noise, y la correlación sube el traffic a both con un único clip unión.

`_save_clip` y `_save_passage_clip` se reemplazan por dobles que crean un
archivo real en el directorio de clips: así se mide qué ventana recibe el
clip unión y qué archivo se borra, sin depender de ffmpeg.
"""

import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.event_manager import EventManager  # noqa: E402
from support import create_events_table, example_config  # noqa: E402

PRE = 2.0
POST = 2.0


class UnifiedWriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = example_config(self.tmp.name)
        cfg["buffer"]["pre_event_seconds"] = PRE
        cfg["buffer"]["post_event_seconds"] = POST
        cfg["triggers"]["cooldown_seconds"] = 0

        # Todo ocurre en el pasado: el post-roll ya existe y no hay sleep.
        self.now = time.time()
        self.base = self.now - 20.0
        car = {"class": "car", "score": 0.7, "cx": 400, "cy": 700, "lane": "left",
               "bbox": [0.6, 0.2, 0.7, 0.3]}
        video = MagicMock()
        video.detections_between.return_value = [car]
        video.recent_detections = [car]
        video.video_buffer = [
            (self.base + i * 0.1, np.zeros((8, 8, 3), dtype=np.uint8)) for i in range(181)
        ]
        audio = MagicMock()
        audio.audio_buffer = [
            (self.base + i * 0.1, np.zeros(16, dtype=np.float32)) for i in range(181)
        ]
        audio.current_db = 40.0
        audio.level_stats.return_value = None
        audio.level_snapshot.return_value = []

        self.em = EventManager(audio, video, config=cfg, base_dir=PROJECT_DIR)
        self.db = cfg["paths"]["database"]
        create_events_table(self.db)

        self.clip_calls = []
        self.fail_clip = set()  # números de llamada a _save_clip que fallan
        self.em._save_clip = self._fake_save_clip
        self.em._save_passage_clip = self._fake_passage_clip

    def tearDown(self):
        self.tmp.cleanup()

    # -- dobles ------------------------------------------------------------
    def _write(self, name: str) -> str:
        path = self.em.clips_dir / name
        path.write_bytes(b"mp4")
        return str(path)

    def _fake_save_clip(self, audio_blocks, video_frames, sample_rate, event_timestamp,
                        mute_audio=False, prefix="event", unique_name=False):
        n = len(self.clip_calls)
        self.clip_calls.append({
            "audio": [ts for ts, _ in audio_blocks],
            "video": [ts for ts, _ in video_frames],
            "mute": mute_audio,
        })
        if n in self.fail_clip:
            raise RuntimeError("ffmpeg falló")
        return self._write(f"{prefix}_{n}.mp4"), 1000 + n

    def _fake_passage_clip(self, passage):
        return self._write(f"passage_{passage['track_id']}.mp4"), 5000, False

    # -- helpers -----------------------------------------------------------
    def _passage(self, enter, exit_, track_id=7, cls="car"):
        passage = {
            "vehicle_class": cls, "lane": "left", "direction": "ltr",
            "speed_kmh": 30.0, "detection_score": 0.6, "track_id": track_id,
            "timestamp_enter": self.base + enter, "timestamp_exit": self.base + exit_,
        }
        self.em._process_passage(passage)
        return passage

    def _trigger(self, at, sound_class="Car", category="vehicle"):
        self.em._process_trigger({
            "type": "peak", "db": 70.0, "timestamp": self.base + at,
            "sound_class": sound_class, "sound_category": category,
            "sound_confidence": 0.8,
        })

    def _rows(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id")]
        finally:
            conn.close()

    # -- casos -------------------------------------------------------------
    def test_pasaje_solo_es_traffic_con_track_id(self):
        self._passage(5.0, 7.0, track_id=42)
        (row,) = self._rows()
        self.assertEqual(row["kind"], "traffic")
        self.assertEqual(row["track_id"], 42)
        self.assertAlmostEqual(row["start_ts"], self.base + 5.0)
        self.assertAlmostEqual(row["end_ts"], self.base + 7.0)
        self.assertEqual(row["db_max"], 40.0)
        self.assertTrue(Path(row["clip_path"]).exists())
        self.assertIsNone(row["peak_ts"])

    def test_ruido_solo_es_noise_con_ventana_pico(self):
        self._trigger(10.0, sound_class="Dog", category="animal")
        (row,) = self._rows()
        self.assertEqual(row["kind"], "noise")
        self.assertAlmostEqual(row["peak_ts"], self.base + 10.0)
        self.assertAlmostEqual(row["start_ts"], self.base + 10.0 - PRE)
        self.assertAlmostEqual(row["end_ts"], self.base + 10.0 + POST)
        self.assertEqual(row["db_max"], 70.0)
        self.assertEqual(row["sound_class"], "Dog")
        self.assertTrue(row["clip_path"])

    def test_upgrade_a_both_un_solo_clip_con_ventana_union(self):
        # Pasaje [3, 9]; ruido en 10 → ventana [8, 12]; unión [3, 12].
        self._passage(3.0, 9.0, track_id=5)
        (traffic,) = self._rows()
        old_clip = Path(traffic["clip_path"])
        self.assertTrue(old_clip.exists())

        self._trigger(10.0)
        rows = self._rows()
        self.assertEqual(len(rows), 1, "el ruido no debe crear una segunda fila")
        row = rows[0]
        self.assertEqual(row["id"], traffic["id"])
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["track_id"], 5)
        self.assertEqual(row["sound_class"], "Car")
        self.assertEqual(row["db_max"], 70.0)
        self.assertAlmostEqual(row["peak_ts"], self.base + 10.0)

        # Una sola llamada a _save_clip, con la ventana unión.
        self.assertEqual(len(self.clip_calls), 1)
        call = self.clip_calls[0]
        self.assertAlmostEqual(min(call["audio"]), self.base + 3.0, places=3)
        self.assertAlmostEqual(max(call["audio"]), self.base + 12.0, places=3)
        self.assertAlmostEqual(min(call["video"]), self.base + 3.0, places=3)
        self.assertAlmostEqual(max(call["video"]), self.base + 12.0, places=3)
        self.assertAlmostEqual(row["start_ts"], self.base + 3.0, places=3)
        self.assertAlmostEqual(row["end_ts"], self.base + 12.0, places=3)

        # Un solo clip: el nuevo existe, el del pasaje se borró.
        self.assertTrue(Path(row["clip_path"]).exists())
        self.assertNotEqual(Path(row["clip_path"]), old_clip)
        self.assertFalse(old_clip.exists(), "el clip previo del pasaje debe borrarse")
        self.assertEqual(row["duration_ms"], 1000)

    def test_union_incluye_el_preroll_del_ruido(self):
        # Ruido antes que el pasaje: la unión arranca en el pre-roll del ruido.
        self._passage(10.0, 12.0)
        self._trigger(9.0)  # ventana [7, 11]
        (row,) = self._rows()
        self.assertEqual(row["kind"], "both")
        call = self.clip_calls[0]
        self.assertAlmostEqual(min(call["audio"]), self.base + 7.0, places=3)
        self.assertAlmostEqual(max(call["audio"]), self.base + 12.0, places=3)

    def test_union_limitada_al_buffer_disponible(self):
        # El buffer termina en base+18: la unión no puede pasarse de ahí.
        self._passage(15.0, 17.5)
        self._trigger(17.0)  # ventana [15, 19]
        (row,) = self._rows()
        self.assertEqual(row["kind"], "both")
        self.assertAlmostEqual(row["end_ts"], self.base + 18.0, places=3)

    def test_fallo_del_clip_union_conserva_el_evento_con_clip_del_ruido(self):
        self._passage(3.0, 9.0)
        (traffic,) = self._rows()
        old_clip = Path(traffic["clip_path"])
        self.fail_clip = {0}
        with self.assertLogs("src.event_manager", level="WARNING"):
            self._trigger(10.0)
        (row,) = self._rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(len(self.clip_calls), 2)
        noise_call = self.clip_calls[1]
        self.assertAlmostEqual(min(noise_call["audio"]), self.base + 8.0, places=3)
        self.assertAlmostEqual(max(noise_call["audio"]), self.base + 12.0, places=3)
        self.assertTrue(Path(row["clip_path"]).exists())
        self.assertAlmostEqual(row["start_ts"], self.base + 8.0)
        self.assertAlmostEqual(row["end_ts"], self.base + 12.0)
        self.assertFalse(old_clip.exists())

    def test_fallan_los_dos_clips_conserva_fila_y_clip_del_pasaje(self):
        self._passage(3.0, 9.0)
        (traffic,) = self._rows()
        self.fail_clip = {0, 1}
        with self.assertLogs("src.event_manager", level="WARNING"):
            self._trigger(10.0)
        (row,) = self._rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["clip_path"], traffic["clip_path"])
        self.assertTrue(Path(row["clip_path"]).exists(), "sin clip nuevo no se borra el viejo")

    def test_candidatos_excluyen_both(self):
        self._passage(3.0, 9.0)
        self._trigger(10.0)
        self._trigger(11.0)
        rows = self._rows()
        self.assertEqual([r["kind"] for r in rows], ["both", "noise"])
        self.assertEqual(len({r["clip_path"] for r in rows}), 2)

    def test_reclamo_atomico_si_la_fila_ya_no_es_traffic(self):
        # Otro escritor subió la fila a both entre la elección en memoria y el UPDATE.
        self._passage(3.0, 9.0)
        (traffic,) = self._rows()
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE events SET kind = 'both', sound_class = 'Otro' WHERE id = ?",
                     (traffic["id"],))
        conn.commit()
        conn.close()
        with self.assertLogs("src.event_manager", level="WARNING"):
            self._trigger(10.0)
        rows = self._rows()
        self.assertEqual([r["kind"] for r in rows], ["both", "noise"])
        self.assertEqual(rows[0]["sound_class"], "Otro", "no se pisan datos de un both ajeno")
        self.assertEqual(rows[0]["clip_path"], traffic["clip_path"])
        self.assertTrue(Path(traffic["clip_path"]).exists())

    def test_pasaje_fijado_no_pierde_su_clip(self):
        self._passage(3.0, 9.0)
        (traffic,) = self._rows()
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE events SET pinned = 1 WHERE id = ?", (traffic["id"],))
        conn.commit()
        conn.close()
        self._trigger(10.0)
        (row,) = self._rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["pinned"], 1)
        self.assertNotEqual(row["clip_path"], traffic["clip_path"])
        self.assertTrue(Path(traffic["clip_path"]).exists())

    def test_clase_distinta_no_correlaciona(self):
        self._passage(3.0, 9.0, cls="truck")
        self._trigger(10.0)
        self.assertEqual([r["kind"] for r in self._rows()], ["traffic", "noise"])

    def test_run_exige_la_tabla_events(self):
        conn = sqlite3.connect(self.db)
        conn.execute("DROP TABLE events")
        conn.commit()
        conn.close()
        with self.assertRaisesRegex(RuntimeError, "migrate_unified"):
            self.em.run()


if __name__ == "__main__":
    unittest.main()
