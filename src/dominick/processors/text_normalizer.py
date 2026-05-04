import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Tuple

import pandas as pd

@dataclass
class ProductTextNormalizer:
    """
    Extracts normalized variables from `product_description`.

    Main outputs:
    - brand_family_norm
    - style_segment_norm

    Usage:
        normalizer = ProductTextNormalizer()
        df = normalizer.transform(df, text_col="product_description")
    """

    unknown_brand: str = "UNKNOWN_BRAND"
    unknown_style: str = "UNKNOWN_STYLE"

    brand_rules: List[Tuple[Pattern[str], str]] = field(init=False)
    style_rules: List[Tuple[Pattern[str], str]] = field(init=False)

    # If no custom rules are provided, initialize with built-in defaults.
    # This keeps the normalizer usable out of the box while still allowing overrides.
    def __post_init__(self) -> None:
        self.brand_rules = self._default_brand_rules()
        self.style_rules = self._default_style_rules()

    # Remove diacritics (e.g., "Á" -> "A") so regex matching is robust to accents.
    @staticmethod
    def _strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in text if not unicodedata.combining(ch))

    # Canonical text normalization used by all downstream extractors:
    # - uppercase
    # - accent stripping
    # - punctuation harmonization
    # - non-alphanumeric cleanup
    # - whitespace collapsing
    # This makes pattern matching stable across noisy product descriptions.
    @classmethod
    def normalize_text(cls, text: Optional[str]) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        text = str(text).upper().strip()
        text = cls._strip_accents(text)

        # Normalizaciones de puntuación frecuentes
        text = text.replace("&", " AND ")
        text = text.replace("/", " ")
        text = text.replace("-", " ")
        text = text.replace("'", "")
        text = text.replace('"', " ")

        # Quitar basura frecuente y colapsar espacios
        text = re.sub(r"[^A-Z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # Brand extraction strategy:
    # 1) try explicit regex dictionary (high precision),
    # 2) fallback to a simple token heuristic when no rule matches.
    # Returns a canonical brand family label.
    def extract_brand_family(self, text: Optional[str]) -> str:
        text_norm = self.normalize_text(text)
        if not text_norm:
            return self.unknown_brand

        for pattern, label in self.brand_rules:
            if pattern.search(text_norm):
                return label

        # Fallback simple: primeras 1-2 palabras no genéricas
        tokens = text_norm.split()
        stopwords = {
            "BEER", "LIGHT", "DRAFT", "DARK", "AMBER", "LAGER", "ALE", "STOUT",
            "PORTER", "NR", "BTL", "BTLS", "BOTTLE", "BOTTLES", "CANS", "CAN",
            "LONGNECK", "LONG", "REGULAR", "GOLDEN", "PREMIUM", "DRY", "ICE",
            "NON", "ALCOHOLIC", "NONALCOHOLIC", "NA", "N", "OZ", "PK", "PACK",
        }
        core_tokens = [t for t in tokens if t not in stopwords]

        if not core_tokens:
            return self.unknown_brand
        if len(core_tokens) == 1:
            return core_tokens[0]
        return f"{core_tokens[0]}_{core_tokens[1]}"

    # Style extraction strategy:
    # - collect all matching style labels,
    # - resolve conflicts with a priority list (more structural categories first).
    # This avoids ambiguous outputs when multiple style keywords coexist.
    def extract_style_segment(self, text: Optional[str]) -> str:
        text_norm = self.normalize_text(text)
        if not text_norm:
            return self.unknown_style

        matched_labels: List[str] = []
        for pattern, label in self.style_rules:
            if pattern.search(text_norm):
                matched_labels.append(label)

        if not matched_labels:
            return self.unknown_style

        # Prioridad: categorías más estructurales primero
        priority = [
            "NON_ALCOHOLIC",
            "CIDER",
            "MALT_BEVERAGE",
            "STOUT",
            "PORTER",
            "PALE_ALE",
            "AMBER_ALE",
            "BROWN_ALE",
            "WHEAT",
            "PILSNER",
            "BOCK",
            "LAGER",
            "LIGHT",
            "ICE",
            "DRAFT",
            "DARK",
            "REGULAR",
            "IMPORT",
            "CRAFT",
        ]
        for label in priority:
            if label in matched_labels:
                return label

        return matched_labels[0]

    # Public API. Adds normalized brand and style columns to DataFrame.
    # Adds normalized text-derived features to the input dataframe:
    # - brand_family_norm
    # - style_segment_norm
    # Returns a copy to avoid mutating the original dataframe in place.
    def transform(self, df: pd.DataFrame, text_col: str = "product_description") -> pd.DataFrame:
        out = df.copy()
        out["brand_family_norm"] = out[text_col].map(self.extract_brand_family)
        out["style_segment_norm"] = out[text_col].map(self.extract_style_segment)
        return out

    # Small helper to centralize regex compilation (single place for future flags/tuning).
    @staticmethod
    def _compile(pattern: str) -> Pattern[str]:
        return re.compile(pattern)

    # Default brand mapping rules:
    # list of (compiled_regex, canonical_label) pairs.
    # Keep labels stable over time because they become model features.
    def _default_brand_rules(self) -> List[Tuple[Pattern[str], str]]:
        rules = [
            (self._compile(r"\bBUDWEISER\b|\bBUD ICE\b|\bBUD\b"), "BUDWEISER"),
            (self._compile(r"\bMICHELOB\b"), "MICHELOB"),
            (self._compile(r"\bMILLER\b"), "MILLER"),
            (self._compile(r"\bMILWAUKEES BEST\b|\bMILWAUKEE\S* BEST\b"), "MILWAUKEES_BEST"),
            (self._compile(r"\bCOORS\b"), "COORS"),
            (self._compile(r"\bKEYSTONE\b"), "KEYSTONE"),
            (self._compile(r"\bHEINEKEN\b"), "HEINEKEN"),
            (self._compile(r"\bAMSTEL\b"), "AMSTEL"),
            (self._compile(r"\bCARLSBERG\b"), "CARLSBERG"),
            (self._compile(r"\bGUINNESS\b|\bGUINESS\b"), "GUINNESS"),
            (self._compile(r"\bMURPHYS\b|\bMURPHYS\b"), "MURPHYS"),
            (self._compile(r"\bPABST\b"), "PABST"),
            (self._compile(r"\bHAMMS\b"), "HAMMS"),
            (self._compile(r"\bLEINENKUGEL\b|\bLEINNKGL\b"), "LEINENKUGEL"),
            (self._compile(r"\bLOWENBRAU\b"), "LOWENBRAU"),
            (self._compile(r"\bGOOSE ISLAND\b"), "GOOSE_ISLAND"),
            (self._compile(r"\bASAHI\b"), "ASAHI"),
            (self._compile(r"\bKIRIN\b"), "KIRIN"),
            (self._compile(r"\bBELLS\b"), "BELLS"),
            (self._compile(r"\bNEW AMSTERDAM\b"), "NEW_AMSTERDAM"),
            (self._compile(r"\bHACKER\b|\bPSCHORR\b"), "HACKER_PSCHORR"),
            (self._compile(r"\bSCHLITZ\b|\bSCHILTZ\b"), "SCHLITZ"),
            (self._compile(r"\bLABATTS\b|\bLABATT\b"), "LABATT"),
            (self._compile(r"\bMOLSON\b"), "MOLSON"),
            (self._compile(r"\bFOSTERS\b|\bFOSTERS\b"), "FOSTERS"),
            (self._compile(r"\bROLLING ROCK\b"), "ROLLING_ROCK"),
            (self._compile(r"\bBLUE MOON\b"), "BLUE_MOON"),
            (self._compile(r"\bKILLIANS\b|\bKILLIANS\b"), "KILLIANS"),
            (self._compile(r"\bSTROHS\b|\bSTROHS\b"), "STROHS"),
            (self._compile(r"\bDOS EQUIS\b"), "DOS_EQUIS"),
            (self._compile(r"\bMOOSEHEAD\b"), "MOOSEHEAD"),
            (self._compile(r"\bANCHOR\b"), "ANCHOR"),
            (self._compile(r"\bOLD STYLE\b|\bOS CLASSIC\b"), "OLD_STYLE"),
            (self._compile(r"\bSPECIAL EXPORT\b"), "SPECIAL_EXPORT"),
            (self._compile(r"\bCORONA\b|\bCORONITA\b"), "CORONA"),
            (self._compile(r"\bMODELO\b"), "MODELO"),
            (self._compile(r"\bPERONI\b"), "PERONI"),
            (self._compile(r"\bWARSTEINER\b"), "WARSTEINER"),
            (self._compile(r"\bBECKS\b|\bBECK\b"), "BECKS"),
            (self._compile(r"\bDUVEL\b"), "DUVEL"),
            (self._compile(r"\bGROLSCH\b"), "GROLSCH"),
            (self._compile(r"\bSIERRA NEVADA\b"), "SIERRA_NEVADA"),
            (self._compile(r"\bBASS\b"), "BASS"),
            (self._compile(r"\bHARP\b"), "HARP"),
            (self._compile(r"\bPETES\b|\bPETES WICKED\b|\bPETE\S\b"), "PETES"),
            (self._compile(r"\bSAMUEL ADAMS\b|\bSAM ADAMS\b"), "SAMUEL_ADAMS"),
            (self._compile(r"\bSAPPORO\b"), "SAPPORO"),
            (self._compile(r"\bNEWCASTLE\b"), "NEWCASTLE"),
            (self._compile(r"\bSPATEN\b"), "SPATEN"),
            (self._compile(r"\bTECATE\b"), "TECATE"),
            (self._compile(r"\bROGUE\b"), "ROGUE"),
            (self._compile(r"\bBADERBRAU\b"), "BADERBRAU"),
            (self._compile(r"\bSHIPYARD\b"), "SHIPYARD"),
            (self._compile(r"\bOREGON\b"), "OREGON"),
            (self._compile(r"\bWOODCHUCK\b"), "WOODCHUCK"),
            (self._compile(r"\bWOODPECKER\b"), "WOODPECKER"),
            (self._compile(r"\bHORNSBYS\b|\bHORNSBYS\b"), "HORNSBYS"),
            (self._compile(r"\bZIMA\b"), "ZIMA"),
            (self._compile(r"\bCLAUSTHALER\b"), "CLAUSTHALER"),
            (self._compile(r"\bODOULS\b"), "ODOULS"),
            (self._compile(r"\bST REGIS\b"), "ST_REGIS"),
            (self._compile(r"\bBUCKLER\b"), "BUCKLER"),
            (self._compile(r"\bARIEL\b"), "ARIEL"),
            (self._compile(r"\bSUTTER HOME\b"), "SUTTER_HOME"),
            (self._compile(r"\bTOSELLI\b"), "TOSELLI"),
            (self._compile(r"\bBEER LIMIT\b"), "BEER_LIMIT"),
        ]
        return rules

    # Default style mapping rules:
    # list of (compiled_regex, canonical_label) pairs for style/segment extraction.
    def _default_style_rules(self) -> List[Tuple[Pattern[str], str]]:
        rules = [
            (self._compile(r"\bNON ALCOHOLIC\b|\bNONALCOHOLIC\b|\bNON ALCH\b|\bN A\b|\bNA\b"), "NON_ALCOHOLIC"),
            (self._compile(r"\bCIDER\b|\bCIDE\b"), "CIDER"),
            (self._compile(r"\bMALT LIQUOR\b|\bMALTBEV\b|\bMALTMALT\b"), "MALT_BEVERAGE"),
            (self._compile(r"\bSTOUT\b"), "STOUT"),
            (self._compile(r"\bPORTER\b"), "PORTER"),
            (self._compile(r"\bPALE ALE\b"), "PALE_ALE"),
            (self._compile(r"\bAMBER ALE\b|\bRED ALE\b|\bIRISH RED\b"), "AMBER_ALE"),
            (self._compile(r"\bBROWN ALE\b|\bNUT BROWN\b"), "BROWN_ALE"),
            (self._compile(r"\bWHEAT\b|\bHEFE\b|\bHEFEWEIZEN\b|\bWHITE\b"), "WHEAT"),
            (self._compile(r"\bPILSNER\b"), "PILSNER"),
            (self._compile(r"\bBOCK\b"), "BOCK"),
            (self._compile(r"\bLAGER\b"), "LAGER"),
            (self._compile(r"\bLIGHT\b|\bLITE\b|\bLT\b"), "LIGHT"),
            (self._compile(r"\bICE\b"), "ICE"),
            (self._compile(r"\bDRAFT\b|\bDRAUGHT\b|\bDRFT\b"), "DRAFT"),
            (self._compile(r"\bDARK\b"), "DARK"),
            (self._compile(r"\bIMPORT\b|\bGERMAN\b|\bBELGIUM\b|\bPOLISH\b|\bKOREAN\b"), "IMPORT"),
            (self._compile(r"\bREGULAR\b|\bREG\b"), "REGULAR"),
        ]
        return rules