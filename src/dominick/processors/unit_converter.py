import re
import numpy as np

class UnitConverter:
    """
    Parses Dominick's raw pack_size_text strings and converts them to liters per UPC.
    Handles four string formats (after normalization):
      1. Multipack with unit  : "6/12OZ", "4/14.9OZ", "12/11OZ"  → n x size x factor
      2. Multipack without unit: "30/12", "6/11.2"               → assumes OZ
      3. Single with unit     : "750ML", "32OZ", "2GAL"          → size x factor
      4. Single without unit  : "32"                             → assumes OZ
    Supported units: OZ (fl oz), ML, GAL / GALLON / GALLONS.
    Returns np.nan for unrecognized formats or missing values.
    Conversion factors:
      1 US fl oz = 0.0295735296 L
      1 US gal   = 3.785411784  L
    Public API:
        run(pack_size_text) -> float  (liters)
    """

    def __init__(self):
        self.OZ_TO_L = 0.0295735296      # US fluid oz -> liters
        self.GAL_TO_L = 3.785411784      # US gallon -> liters

    def _normalize_pack_size(self, s: str) -> str:
        """Normalizes string of pack_size for parsing."""
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
    
    def _fix_missing_decimal_oz(self, raw_size: str, size: float, unit: str) -> float:
        """
        Heuristic to recover a missing decimal point in OZ pack sizes.
        Dominick's beer data occasionally stores sizes like "120", "259", "661"
        when the intended value is "12.0", "25.9", "66.1" oz. The decimal point
        was dropped during data entry. We detect this when unit==OZ, the raw
        string is an integer >= 100, and has no '.', then divide by 10.
        Examples: "120" to 12.0 oz, "259" to 25.9 oz, "661" to 66.1 oz.
        """
        if unit != "OZ":
            return size
        if "." in raw_size:
            return size
        # raw_size is like "120", "259", "661"
        if raw_size.isdigit() and int(raw_size) >= 100:
            return float(raw_size) / 10.0
        return size

    def run(self, pack_size_text: str) -> float:
        """
        Public API. Converts pack_size_text to liters per UPC.
        """
        if pack_size_text is None or (isinstance(pack_size_text, float) and np.isnan(pack_size_text)):
            return np.nan
        # Normalize the pack_size_text string.
        s = self._normalize_pack_size(str(pack_size_text))
        if not s:
            return np.nan

        # 1) multipack with unit: N/SIZEUNIT  (6/12OZ, 4/14.9OZ, 12/11OZ)
        m = re.match(r'^(?P<n>\d+)\/(?P<size>\d+(?:\.\d+)?)(?P<unit>[A-Z]+)$', s)
        if m:
            n = int(m.group("n"))
            raw_size = m.group("size")
            size = float(raw_size)
            unit = m.group("unit")

            # Heuristic decimal lost in OZ
            size = self._fix_missing_decimal_oz(raw_size, size, unit)
            # Convert to liters.
            if unit == "ML":
                return n * (size / 1000.0)
            if unit == "OZ":
                return n * (size * self.OZ_TO_L)
            if unit in ("GAL", "GALLON", "GALLONS"):
                return n * (size * self.GAL_TO_L)
            return np.nan

        # 2) multipack without unit: N/SIZE  -> assume OZ (30/12, 6/11.2, 15/120, 12/11)
        m = re.match(r'^(?P<n>\d+)\/(?P<size>\d+(?:\.\d+)?)$', s)
        if m:
            n = int(m.group("n"))
            raw_size = m.group("size")
            size = float(raw_size)

            # Assume OZ + heuristic for lost decimal
            size = self._fix_missing_decimal_oz(raw_size, size, "OZ")
            return n * (size * self.OZ_TO_L)

        # 3) single with unit: SIZEUNIT (750ML, 32OZ, 259OZ)
        m = re.match(r'^(?P<size>\d+(?:\.\d+)?)(?P<unit>[A-Z]+)$', s)
        if m:
            raw_size = m.group("size")
            size = float(raw_size)
            unit = m.group("unit")

            size = self._fix_missing_decimal_oz(raw_size, size, unit)

            if unit == "ML":
                return size / 1000.0
            if unit == "OZ":
                return size * self.OZ_TO_L
            if unit in ("GAL", "GALLON", "GALLONS"):
                return size * self.GAL_TO_L
            return np.nan

        # 4) single without unit: SIZE -> assume OZ (if it appears)
        m = re.match(r'^(?P<size>\d+(?:\.\d+)?)$', s)
        if m:
            raw_size = m.group("size")
            size = float(raw_size)
            size = self._fix_missing_decimal_oz(raw_size, size, "OZ")
            return size * self.OZ_TO_L

        return np.nan