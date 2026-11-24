"""
Horario semanal de una sección de disparadores (ruido o tráfico).

`{enabled, start, end, days}`: con `enabled: false` la sección está siempre
activa. `start`/`end` son `HH:MM` en hora local; si `start > end` la ventana
cruza la medianoche (p. ej. 22:00–07:00). `days` son días ISO (1 = lunes …
7 = domingo); vacío = todos. En una ventana nocturna el día es el de inicio:
con `days: [5]`, 22:00–07:00 cubre del viernes 22:00 al sábado 07:00.

El fin es exclusivo: 22:00–07:00 está activa a las 06:59 y no a las 07:00.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta

DEFAULT_START = "22:00"
DEFAULT_END = "07:00"


def _parse_hhmm(value, name: str) -> dtime:
    if not isinstance(value, str):
        raise ValueError(f"{name} debe ser HH:MM (recibido: {value!r})")
    parts = value.split(":")
    if len(parts) != 2 or not all(p.isdigit() and len(p) == 2 for p in parts):
        raise ValueError(f"{name} debe ser HH:MM (recibido: {value!r})")
    hour, minute = int(parts[0]), int(parts[1])
    if hour > 23 or minute > 59:
        raise ValueError(f"{name} fuera de rango (recibido: {value!r})")
    return dtime(hour, minute)


@dataclass(frozen=True)
class Schedule:
    enabled: bool = False
    start: dtime = field(default_factory=lambda: _parse_hhmm(DEFAULT_START, "start"))
    end: dtime = field(default_factory=lambda: _parse_hhmm(DEFAULT_END, "end"))
    days: frozenset[int] = frozenset()

    @classmethod
    def parse(cls, raw, name: str = "schedule") -> "Schedule":
        """Valida y construye; `None` o ausente = siempre activo."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"{name} debe ser una sección")
        unknown = set(raw) - {"enabled", "start", "end", "days"}
        if unknown:
            raise ValueError(f"{name}: claves desconocidas {sorted(unknown)}")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"{name}.enabled debe ser booleano (recibido: {enabled!r})")
        start = _parse_hhmm(raw.get("start", DEFAULT_START), f"{name}.start")
        end = _parse_hhmm(raw.get("end", DEFAULT_END), f"{name}.end")
        if start == end:
            raise ValueError(f"{name}: start y end iguales ({raw.get('start')}) dejan la ventana vacía")
        days = raw.get("days") or []
        if not isinstance(days, list) or any(
            isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= 7 for d in days
        ):
            raise ValueError(f"{name}.days debe ser una lista de días 1–7 (recibido: {days!r})")
        return cls(enabled, start, end, frozenset(days))

    def is_active(self, now: datetime) -> bool:
        if not self.enabled:
            return True
        clock = now.time()
        if self.start < self.end:
            inside = self.start <= clock < self.end
            day = now
        elif clock >= self.start:
            inside, day = True, now
        elif clock < self.end:
            inside, day = True, now - timedelta(days=1)
        else:
            inside, day = False, now
        return inside and (not self.days or day.isoweekday() in self.days)

    def is_active_at(self, epoch: float) -> bool:
        """`is_active` para un timestamp epoch, en hora local del host."""
        return self.is_active(datetime.fromtimestamp(epoch))
