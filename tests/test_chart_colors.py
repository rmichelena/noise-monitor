#!/usr/bin/env python3
"""El gráfico de audio usa las mismas bandas de color que el overlay."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.app import DB_BANDS, db_color  # noqa: E402


class ChartColorsTest(unittest.TestCase):
    def test_bandas(self):
        self.assertEqual(db_color(20), "#50c850")
        self.assertEqual(db_color(45), "#ffd23f")
        self.assertEqual(db_color(55), "#ffb020")
        self.assertEqual(db_color(65), "#ff5c5c")
        # borde superior
        self.assertEqual(db_color(40), DB_BANDS[0][1])
        self.assertEqual(db_color(41), DB_BANDS[1][1])


if __name__ == "__main__":
    unittest.main()
