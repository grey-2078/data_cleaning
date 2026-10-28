import itertools
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.config import (
    CITY_STATE_ZIP_REGEX,
    MISSING_VALUE_PLACEHOLDERS,
    SEMANTIC_DUPLICATE_SUBSET,
    WHITESPACE_REGEX,
)

logger = logging.getLogger(__name__)


class DataQualityCleaner:
    """Базовый обзор датасета: нормализация пропусков, поиск сдвигов между колонками и чистка дубликатов."""

    @staticmethod
    def generate_column_audit_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Собирает сводную таблицу по колонкам: типы данных, доля пропусков,
        число уникальных значений и несоответствия содержимого названию колонки.
        """
        records = []
        for col in df.columns:
            series = df[col]
            n_missing = series.isna().sum()
            pct_missing = (n_missing / len(df)) * 100
            n_unique = series.nunique(dropna=True)
            sample_val = series.dropna().iloc[0] if n_unique > 0 else np.nan

            issues = []
            if n_unique <= 1:
                issues.append("Константная колонка (1 значение)")
            if pct_missing > 80:
                issues.append("Много пропусков (>80%)")

            # Проверяем, не уехали ли адреса формата "City, ST" в соседние текстовые колонки
            if col != "location" and series.dtype == "object":
                sample_str = series.dropna().astype(str).head(500)
                loc_like_ratio = sample_str.str.match(CITY_STATE_ZIP_REGEX).mean()
                if loc_like_ratio > 0.05:
                    issues.append(f"Сдвиг данных: содержит локации ({loc_like_ratio:.1%})")

            records.append({
                "column": col,
                "dtype": str(series.dtype),
                "missing_count": n_missing,
                "missing_pct": round(pct_missing, 2),
                "unique_count": n_unique,
                "sample_value": str(sample_val)[:50],
                "diagnosed_issues": "; ".join(issues) if issues else "OK",
            })

        return pd.DataFrame(records)

    @staticmethod
    def find_column_intersections(df: pd.DataFrame, min_overlap: int = 5) -> pd.DataFrame:
        """
        Ищет пересечения уникальных значений между парами строковых колонок через множества.
        Помогает отловить сдвиги при парсинге CSV (например, когда локация попадает в organization или sector).
        """
        obj_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c != "uniq_id"]
        col_sets: Dict[str, set] = {
            col: set(df[col].dropna().astype(str).str.strip().unique()) - {""}
            for col in obj_cols
        }

        overlaps = []
        for col_a, col_b in itertools.combinations(obj_cols, 2):
            shared = col_sets[col_a].intersection(col_sets[col_b])
            if len(shared) >= min_overlap:
                overlaps.append({
                    "column_a": col_a,
                    "column_b": col_b,
                    "shared_unique_values": len(shared),
                    "examples": list(shared)[:5],
                })

        overlap_df = pd.DataFrame(overlaps)
        if not overlap_df.empty:
            overlap_df = overlap_df.sort_values(by="shared_unique_values", ascending=False).reset_index(drop=True)
        return overlap_df

    @staticmethod
    def fix_organization_location_swaps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Исправляет строки, где в 'organization' записан стандартный адрес 'City, ST Zip':
        переносит адрес в 'location' (если там пусто или мусор) и затирает ошибочное значение.
        """
        if "organization" not in df.columns or "location" not in df.columns:
            return df

        df_clean = df.copy()
        org_is_loc = df_clean["organization"].astype(str).str.match(CITY_STATE_ZIP_REGEX, na=False)
        loc_is_missing_or_bad = ~df_clean["location"].astype(str).str.match(CITY_STATE_ZIP_REGEX, na=False)

        swap_mask = org_is_loc & loc_is_missing_or_bad
        if swap_mask.any():
            df_clean.loc[swap_mask, "location"] = df_clean.loc[swap_mask, "organization"]
            df_clean.loc[swap_mask, "organization"] = np.nan
            logger.info(f"Исправлено {swap_mask.sum()} строк со сдвигом локации в 'organization'.")
        else:
            df_clean.loc[org_is_loc, "organization"] = np.nan

        return df_clean

    @staticmethod
    def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Заменяет пустые строки, пробелы и текстовые плейсхолдеры ('N/A', 'none', '-') на np.nan."""
        df_clean = df.copy()
        obj_cols = df_clean.select_dtypes(include=["object", "string"]).columns

        for col in obj_cols:
            # Убираем пробелы по краям и приводим к строковому типу pandas
            s = df_clean[col].astype("string").str.strip()
            mask = (
                s.str.lower().isin(MISSING_VALUE_PLACEHOLDERS)
                | s.str.match(WHITESPACE_REGEX, na=False)
                | (s == "")
                | s.isna()
            )
            df_clean[col] = s.mask(mask, np.nan).astype(object)

        logger.info("Скрытые пробелы и текстовые плейсхолдеры приведены к NaN.")
        return df_clean

    @staticmethod
    def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Находит и удаляет константные колонки (где всего одно уникальное значение без учета NaN)."""
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
        if constant_cols:
            logger.info(f"Удаляем константные колонки: {constant_cols}")
            df = df.drop(columns=constant_cols)
        return df, constant_cols

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Удаляет полные дубликаты, повторы по uniq_id и семантические дубликаты вакансий."""
        initial_len = len(df)

        # 1. Полные дубликаты строк
        df = df.drop_duplicates()

        # 2. Уникальность первичного ключа uniq_id
        if "uniq_id" in df.columns:
            df = df.drop_duplicates(subset=["uniq_id"], keep="first")

        # 3. Семантические дубликаты (одна и та же вакансия с разными id/ссылками)
        valid_subset = [c for c in SEMANTIC_DUPLICATE_SUBSET if c in df.columns]
        if valid_subset:
            df = df.drop_duplicates(subset=valid_subset, keep="first")

        logger.info(f"Удалено {initial_len - len(df)} дубликатов ({initial_len} -> {len(df)}).")
        return df.reset_index(drop=True)