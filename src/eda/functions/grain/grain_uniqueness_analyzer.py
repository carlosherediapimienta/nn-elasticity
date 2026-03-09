import pandas as pd


class GrainUniquenessAnalyzer:
    """
    Valida que un conjunto de columnas define el grano (unicidad) del DataFrame.
    Public API: run().
    """

    def run(self, df: pd.DataFrame, grain_cols: list[str]) -> pd.DataFrame:
        """
        Args:
            df: DataFrame con los datos.
            grain_cols: columnas que definen el grano, por ejemplo
                        ["week_id", "store_code", "upc_code"].

        Returns:
            DataFrame de una sola fila con:
                - grain_cols: columnas del grano (como string separado por comas)
                - n_rows: número total de filas
                - n_unique_keys: número de combinaciones únicas del grano
                - n_duplicate_keys: número de claves de grano con más de 1 fila
                - pct_duplicate_keys: % de claves de grano duplicadas
                - n_rows_with_missing_in_grain: nº de filas con al menos un NaN en el grano
                - is_unique_grain: bool, True si el grano es perfectamente único y sin NaNs
        """
        missing_cols = [c for c in grain_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas del grano en el DataFrame: {missing_cols}")

        grain_df = df[grain_cols]

        # Filas con algún NaN en el grano
        n_rows_with_missing = int(grain_df.isna().any(axis=1).sum())

        n_rows = int(len(df))

        # Conteo de combinaciones de grano (incluyendo NaNs si los hubiera)
        key_counts = (
            grain_df
            .groupby(grain_cols, dropna=False)
            .size()
        )

        n_unique_keys = int(key_counts.shape[0])
        n_duplicate_keys = int((key_counts > 1).sum())

        if n_unique_keys > 0:
            pct_duplicate_keys = round(n_duplicate_keys / n_unique_keys * 100, 4)
        else:
            pct_duplicate_keys = 0.0

        is_unique_grain = (n_rows == n_unique_keys) and (n_rows_with_missing == 0)

        result = pd.DataFrame(
            [{
                "grain_cols": ", ".join(grain_cols),
                "n_rows": n_rows,
                "n_unique_keys": n_unique_keys,
                "n_duplicate_keys": n_duplicate_keys,
                "pct_duplicate_keys": pct_duplicate_keys,
                "n_rows_with_missing_in_grain": n_rows_with_missing,
                "is_unique_grain": is_unique_grain,
            }]
        )

        return result