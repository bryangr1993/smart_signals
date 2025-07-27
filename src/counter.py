"""
counter.py
==========

Contador instantáneo de vehículos en un polígono ROI.

Mejoras clave
-------------
* **min_hits** : nº mínimo de frames consecutivos dentro del ROI antes de aceptar
  un ID → evita falsos positivos y recuentos por oclusiones muy breves.
* **max_miss** : nº de frames sucesivos fuera del ROI antes de retirar un ID →
  evita descontar vehículos por micropérdidas de tracking.
* Métodos `start_frame()` / `finalize_frame()` para un manejo limpio por frame.

Uso típico en main.py
---------------------
counter = VehicleCounter("config/roi.json", min_hits=3, max_miss=7)

while True:
    counter.start_frame()
    ... llamar counter.update(id, bbox) por cada detección ...
    counter.finalize_frame()
    current = counter.current_count
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
from shapely.geometry import Point, Polygon


class VehicleCounter:
    """
    Cuenta vehículos presentes dentro de un ROI en tiempo real.

    Parameters
    ----------
    roi_json : str | Path
        Ruta al JSON con el polígono ROI: {"roi": [[x1, y1], [x2, y2], ...]}
    min_hits : int, default=3
        Frames consecutivos dentro antes de aceptar el ID (anti parpadeos).
    max_miss : int, default=7
        Frames consecutivos fuera antes de retirarlo (tolerancia a pérdidas).
    """

    def __init__(
        self,
        roi_json: str | Path,
        *,
        min_hits: int = 3,
        max_miss: int = 7,
    ) -> None:
        self.roi: Polygon = Polygon(self._load_roi(roi_json))
        self.min_hits = min_hits
        self.max_miss = max_miss

        # Estados internos
        self._hits: Dict[int, int] = {}      # id -> frames dentro consecutivos
        self._miss: Dict[int, int] = {}      # id -> frames fuera consecutivos
        self._ids_inside: Set[int] = set()   # IDs aceptados y vigentes

        self._seen_this_frame: Set[int] = set()  # IDs procesados en el frame

    # ------------------------------------------------------------------ #
    # Ciclo por frame                                                    #
    # ------------------------------------------------------------------ #
    def start_frame(self) -> None:
        """Llamar al **inicio** de cada frame para limpiar la lista temporal."""
        self._seen_this_frame.clear()

    def update(self, track_id: int, bbox: Tuple[int, int, int, int]) -> None:
        """
        Procesa UNA detección del frame actual.

        Parameters
        ----------
        track_id : int
            ID que provee el tracker (ByteTrack, DeepSORT, etc.).
        bbox : tuple[int, int, int, int]
            Cuadro (x1, y1, x2, y2) de la detección.

        Notes
        -----
        • Se debe llamar una vez por cada detección del frame.
        """
        self._seen_this_frame.add(track_id)

        centroid = self._centroid(bbox)
        inside = self.roi.contains(Point(centroid))

        if inside:
            # Incrementa hits y borra miss
            self._hits[track_id] = self._hits.get(track_id, 0) + 1
            self._miss.pop(track_id, None)

            # Acepta vehículo cuando supera min_hits
            if self._hits[track_id] >= self.min_hits:
                self._ids_inside.add(track_id)

        else:
            # Si estaba dentro, comienza a contar misses
            if track_id in self._ids_inside:
                self._miss[track_id] = self._miss.get(track_id, 0) + 1
                if self._miss[track_id] > self.max_miss:
                    self._remove_id(track_id)

    def finalize_frame(self) -> None:
        """
        Llamar al **final** de cada frame**.

        Para los IDs que estaban dentro pero NO aparecieron este frame,
        incrementa su contador de miss y decide si retirarlos.
        """
        for vid in list(self._ids_inside):
            if vid not in self._seen_this_frame:
                self._miss[vid] = self._miss.get(vid, 0) + 1
                if self._miss[vid] > self.max_miss:
                    self._remove_id(vid)

    # ------------------------------------------------------------------ #
    # API pública sencilla                                               #
    # ------------------------------------------------------------------ #
    @property
    def current_count(self) -> int:
        """Número de vehículos vigentes dentro del ROI."""
        return len(self._ids_inside)

    @property
    def ids_inside(self) -> Set[int]:
        """Copia de IDs actualmente dentro (solo lectura)."""
        return set(self._ids_inside)

    def clear(self) -> None:
        """Reinicia todos los estados (útil al procesar otro vídeo)."""
        self._hits.clear()
        self._miss.clear()
        self._ids_inside.clear()
        self._seen_this_frame.clear()

    # ------------------------------------------------------------------ #
    # Métodos privados                                                   #
    # ------------------------------------------------------------------ #
    def _remove_id(self, track_id: int) -> None:
        """Elimina por completo un ID de todos los contadores."""
        self._ids_inside.discard(track_id)
        self._hits.pop(track_id, None)
        self._miss.pop(track_id, None)

    @staticmethod
    def _load_roi(path: str | Path) -> np.ndarray:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return np.asarray(data["roi"], dtype=float)

    @staticmethod
    def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
