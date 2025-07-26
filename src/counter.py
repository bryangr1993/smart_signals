"""
counter.py  ·  Conteo de vehículos con ROI + línea de cruce.
"""

import json
from shapely.geometry import Point, Polygon

class VehicleCounter:
    def __init__(self, roi_json: str, line_coords: tuple):
        # roi_json: ruta a config/roi.json
        # line_coords: ((x1,y1), (x2,y2))
        self.roi = Polygon(json.load(open(roi_json))["roi"])
        self.line = line_coords
        self.memory = {}              # id -> lado anterior (True/False)
        self.in_count = 0
        self.out_count = 0

    # ---------- helpers internos ----------
    def _centroid(self, box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2, (y1+y2)/2)

    def _side(self, point):
        """Devuelve True si el punto está arriba/izq de la línea, False abajo/der."""
        (x1,y1),(x2,y2) = self.line
        return (x2-x1)*(point[1]-y1) - (y2-y1)*(point[0]-x1) > 0

    # ---------- API pública ----------
    def update(self, track_id: int, box: tuple):
        """
        Recibe:
        • track_id : ID del objeto que da YOLO+ByteTrack
        • box      : (x1,y1,x2,y2) del frame actual
        """
        c = self._centroid(box)
        if not self.roi.contains(Point(c)):           # fuera de ROI
            return

        side_now = self._side(c)
        side_prev = self.memory.get(track_id)

        if side_prev is not None and side_prev != side_now:
            if side_now:
                self.in_count  += 1
            else:
                self.out_count += 1
        self.memory[track_id] = side_now
