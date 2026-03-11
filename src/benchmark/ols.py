import pandas as pd
from typing import Dict, Any
from .builder import DesignMatrixBuilder
from .model import LogLogElasticityModel

class ElasticityPipeline:
    """Orquesta el flujo sin asumir responsabilidades de carga o modelado."""

    def __init__(
        self,
        matrix_builder: DesignMatrixBuilder,
        model: LogLogElasticityModel,
    ) -> None:
        self.matrix_builder = matrix_builder
        self.model = model

    def run(self, df: pd.DataFrame ) -> Dict[str, Any]:
        X, y = self.matrix_builder.build(df)
        self.model.fit(X, y)

        return {
            "elasticity": self.model.elasticity(),
            "metrics": self.model.metrics(),
            "summary": self.model.summary_text(),
        }