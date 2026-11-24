"""
Qué disparadores generan eventos, y cuándo (`triggers.*` y `tracker.record_classes`).

Todas las claves son opcionales; ausentes = comportamiento anterior (ruido y
tráfico habilitados todo el día, todas las clases de vehículo).

- Ruido apagado o fuera de horario: AudioMonitor no emite disparos. La
  captura, el buffer y `audio_levels` siguen.
- Tráfico apagado, fuera de horario, clase fuera de `record_classes` o
  `traffic_only_with_noise`: el pasaje no se guarda como `traffic`, pero el
  tracker sigue y el pasaje queda en memoria para correlacionar. Si un ruido
  lo reclama, el evento se guarda como `both`.
- `noise_only_with_traffic`: un ruido sin pasaje correlacionado se descarta.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schedule import Schedule
from .vehicle_tracker import VEHICLE_CLASSES

_BOOL_KEYS = ("noise_enabled", "noise_only_with_traffic", "traffic_enabled", "traffic_only_with_noise")


@dataclass(frozen=True)
class TriggerPolicy:
    noise_enabled: bool = True
    noise_only_with_traffic: bool = False
    noise_schedule: Schedule = Schedule()
    traffic_enabled: bool = True
    traffic_only_with_noise: bool = False
    traffic_schedule: Schedule = Schedule()
    record_classes: frozenset[str] = frozenset(VEHICLE_CLASSES)

    @classmethod
    def from_config(cls, config: dict) -> "TriggerPolicy":
        """Lanza ValueError nombrando la clave si algo es inválido."""
        triggers = config.get("triggers") or {}
        values = {}
        for key in _BOOL_KEYS:
            value = triggers.get(key, cls.__dataclass_fields__[key].default)
            if not isinstance(value, bool):
                raise ValueError(f"triggers.{key} debe ser booleano (recibido: {value!r})")
            values[key] = value
        noise_schedule = Schedule.parse(triggers.get("noise_schedule"), "triggers.noise_schedule")
        traffic_schedule = Schedule.parse(triggers.get("traffic_schedule"), "triggers.traffic_schedule")

        tracker = config.get("tracker") or {}
        raw_classes = tracker.get("record_classes")
        if raw_classes is None:
            classes = frozenset(VEHICLE_CLASSES)
        else:
            if not isinstance(raw_classes, list) or not all(isinstance(c, str) for c in raw_classes):
                raise ValueError(f"tracker.record_classes debe ser una lista (recibido: {raw_classes!r})")
            unknown = set(raw_classes) - VEHICLE_CLASSES
            if unknown:
                raise ValueError(
                    f"tracker.record_classes: clases desconocidas {sorted(unknown)} "
                    f"(válidas: {sorted(VEHICLE_CLASSES)})"
                )
            classes = frozenset(raw_classes)

        # Un flag que exige la otra sección no puede convivir con ella apagada:
        # no se guardaría ningún evento de la sección que sí está habilitada.
        if values["noise_enabled"] and values["noise_only_with_traffic"] and not values["traffic_enabled"]:
            raise ValueError(
                "triggers.noise_only_with_traffic requiere triggers.traffic_enabled: "
                "sin tráfico no habría ningún pasaje con el que correlacionar"
            )
        if values["traffic_enabled"] and values["traffic_only_with_noise"] and not values["noise_enabled"]:
            raise ValueError(
                "triggers.traffic_only_with_noise requiere triggers.noise_enabled: "
                "sin ruido ningún pasaje se guardaría"
            )
        return cls(
            noise_schedule=noise_schedule, traffic_schedule=traffic_schedule,
            record_classes=classes, **values,
        )

    def noise_active(self, epoch: float) -> bool:
        return self.noise_enabled and self.noise_schedule.is_active_at(epoch)

    def traffic_active(self, epoch: float) -> bool:
        return self.traffic_enabled and self.traffic_schedule.is_active_at(epoch)

    def records_passage(self, vehicle_class: str, epoch: float) -> bool:
        """Si un pasaje se guarda como `traffic` por sí mismo."""
        return (
            self.traffic_active(epoch)
            and not self.traffic_only_with_noise
            and vehicle_class in self.record_classes
        )
