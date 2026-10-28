import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from src.config import CITY_STATE_ZIP_REGEX, STATE_CODE_REGEX, ZIP_CODE_REGEX

logger = logging.getLogger(__name__)


class LocationParser:
    """Классифицирует сырые строки локаций и вытаскивает город, код штата и почтовый индекс."""

    @staticmethod
    def classify_location(loc: Optional[str]) -> str:
        """Определяет категорию формата локации для анализа качества данных и построения графиков."""
        if pd.isna(loc) or not str(loc).strip():
            return "Missing"

        loc_str = str(loc).strip()
        if CITY_STATE_ZIP_REGEX.match(loc_str):
            return "Standard (City, ST [Zip])"
        if ZIP_CODE_REGEX.search(loc_str) and STATE_CODE_REGEX.search(loc_str):
            return "Unstructured with State & Zip"
        if STATE_CODE_REGEX.search(loc_str):
            return "Contains State Code"
        if ZIP_CODE_REGEX.search(loc_str):
            return "Zip Only / Numeric"
        if len(loc_str) > 40:
            return "Description / Noise (Long String)"
        return "Other / Non-Standard"

    @staticmethod
    def extract_location_parts(loc: Optional[str]) -> Dict[str, Optional[str]]:
        """Разбирает строку локации на структурированные поля: city, state (2 буквы) и zip_code."""
        result = {"city": np.nan, "state": np.nan, "zip_code": np.nan}
        if pd.isna(loc) or not str(loc).strip():
            return result

        loc_str = str(loc).strip()

        # Быстрый путь: строка сразу подходит под стандартный шаблон "City, ST 12345" или "City, ST"
        match = CITY_STATE_ZIP_REGEX.match(loc_str)
        if match:
            result["city"] = match.group("city").strip()
            result["state"] = match.group("state")
            result["zip_code"] = match.group("zip") if match.group("zip") else np.nan
            return result

        # Фоллбэк: вытаскиваем индекс и код штата по отдельным регуляркам
        zip_match = ZIP_CODE_REGEX.search(loc_str)
        if zip_match:
            result["zip_code"] = zip_match.group(1)
            loc_str = loc_str.replace(zip_match.group(0), "").strip(" ,-")

        state_match = STATE_CODE_REGEX.search(loc_str)
        if state_match:
            result["state"] = state_match.group(1)
            # Все, что идет до кода штата (если это не длинное текстовое описание), берем как город
            city_candidate = loc_str[: state_match.start()].strip(" ,-")
            if city_candidate and len(city_candidate) <= 35:
                result["city"] = city_candidate

        return result

    def transform(self, df: pd.DataFrame, col: str = "location") -> pd.DataFrame:
        """Добавляет в датафрейм формат локации и распарсенные колонки city, state, zip_code."""
        if col not in df.columns:
            return df

        df_out = df.copy()
        df_out["location_format"] = df_out[col].apply(self.classify_location)

        extracted = df_out[col].apply(self.extract_location_parts).apply(pd.Series)
        df_out[["city", "state", "zip_code"]] = extracted

        logger.info("Извлечены структурированные признаки локации: city, state, zip_code.")
        return df_out