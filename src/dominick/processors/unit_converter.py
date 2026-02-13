import re
import numpy as np

class UnitConverter:
    """
    Convierte pack_size_text a litros por UPC.
    API pública: run().
    """

    def __init__(self):
        self.OZ_TO_L = 0.0295735296      # US fluid oz -> liters
        self.GAL_TO_L = 3.785411784      # US gallon -> liters

    def _normalize_pack_size(self, s: str) -> str:
        """Normaliza string de pack_size para parsing."""
        s = s.strip().upper()
        s = s.replace(" ", "").replace("OZ.", "OZ")
        if s.endswith("."):
            s = s[:-1]

        # "16.9O" -> "16.9OZ"
        s = re.sub(r'(?<=\d)O$', 'OZ', s)
        # "12/12O" -> "12/12OZ"
        s = re.sub(r'/(\d+(?:\.\d+)?)O$', r'/\1OZ', s)
        # "5.16GA" -> "5.16GAL"
        s = s.replace("GA", "GAL")
        return s

    def run(self, pack_size_text: str) -> float:
        """
        API pública. Convierte pack_size_text a litros por UPC.
        Retorna NaN si no puede parsear.
        """
        if pack_size_text is None or (isinstance(pack_size_text, float) and np.isnan(pack_size_text)):
            return np.nan

        s = self._normalize_pack_size(str(pack_size_text))
        if not s:
            return np.nan

        # multipack: N/SIZEUNIT (e.g., 6/12OZ)
        m = re.match(r'^(?P<n>\d+)\/(?P<size>\d+(?:\.\d+)?)(?P<unit>[A-Z]+)$', s)
        if m:
            n = int(m.group("n"))
            size = float(m.group("size"))
            unit = m.group("unit")
            if unit == "ML":
                return n * (size / 1000.0)
            if unit == "OZ":
                return n * (size * self.OZ_TO_L)
            if unit in ("GAL", "GALLON", "GALLONS"):
                return n * (size * self.GAL_TO_L)
            return np.nan

        # single: SIZEUNIT (e.g., 750ML, 32OZ)
        m = re.match(r'^(?P<size>\d+(?:\.\d+)?)(?P<unit>[A-Z]+)$', s)
        if m:
            size = float(m.group("size"))
            unit = m.group("unit")
            if unit == "ML":
                return size / 1000.0
            if unit == "OZ":
                return size * self.OZ_TO_L
            if unit in ("GAL", "GALLON", "GALLONS"):
                return size * self.GAL_TO_L
            return np.nan

        return np.nan