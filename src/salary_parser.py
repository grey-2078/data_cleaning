import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from src.config import SALARY_NUMBER_REGEX, SALARY_PERIOD_KEYWORDS

logger = logging.getLogger(__name__)


class SalaryParser:
    """Классифицирует форматы зарплат, парсит числовые границы и восстанавливает пропущенные периоды выплат."""

    @staticmethod
    def classify_salary_format(sal: Optional[str]) -> str:
        """Разбивает сырые строки зарплаты по категориям (диапазон/одно число, с периодом/без периода, текст)."""
        if pd.isna(sal) or not str(sal).strip():
            return "Missing"

        sal_lower = str(sal).lower()
        has_period = any(
            kw in sal_lower for kws in SALARY_PERIOD_KEYWORDS.values() for kw in kws
        )
        matches = SALARY_NUMBER_REGEX.findall(sal_lower)
        num_count = len([m for m in matches if float(m[0].replace(",", "")) > 0])

        if num_count >= 2 and has_period:
            return "Range with Period"
        if num_count >= 2 and not has_period:
            return "Range without Period"
        if num_count == 1 and has_period:
            return "Single Value with Period"
        if num_count == 1 and not has_period:
            return "Single Value without Period"
        return "Text / Non-Numeric Noise"

    @staticmethod
    def parse_salary_string(sal: Optional[str]) -> Dict[str, Optional[float | str]]:
        """Вытаскивает min, max, среднее значение зарплаты и явно указанный период (year, hour и т.д.)."""
        res = {
            "salary_min": np.nan,
            "salary_max": np.nan,
            "salary_mean": np.nan,
            "salary_period": np.nan,
        }
        if pd.isna(sal) or not str(sal).strip():
            return res

        sal_lower = str(sal).lower()

        # 1. Ищем явное указание периода выплат по ключевым словам
        for period, keywords in SALARY_PERIOD_KEYWORDS.items():
            if any(kw in sal_lower for kw in keywords):
                res["salary_period"] = period
                break

        # 2. Вытаскиваем все числа (учитываем суффикс 'k' = * 1000)
        matches = SALARY_NUMBER_REGEX.findall(sal_lower)
        numbers = []
        for num_str, k_suffix in matches:
            val = float(num_str.replace(",", ""))
            if k_suffix:
                val *= 1_000.0
            if val > 0:
                numbers.append(val)

        if not numbers:
            return res

        # Если указана вилка — берем первые две границы, иначе min = max
        low = numbers[0]
        high = numbers[1] if len(numbers) > 1 and numbers[1] >= low else low

        res["salary_min"] = low
        res["salary_max"] = high
        res["salary_mean"] = (low + high) / 2.0
        return res

    @staticmethod
    def impute_missing_periods(df: pd.DataFrame) -> pd.DataFrame:
        """
        Импутирует пропущенный salary_period для строк, где число парсится, а период не указан:
        сравниваем значение с медианными зарплатами по известным периодам (в лог-шкале).
        """
        df_out = df.copy()
        known_mask = df_out["salary_mean"].notna() & df_out["salary_period"].notna()

        if not known_mask.any():
            return df_out

        # Считаем эталонные медианы для каждого известного периода (hour, year, month и т.д.)
        period_medians = (
            df_out.loc[known_mask]
            .groupby("salary_period")["salary_mean"]
            .median()
            .to_dict()
        )
        logger.info(f"Медианные зарплаты по периодам: {period_medians}")

        def _assign_closest_period(val: float) -> str:
            # Берем период с минимальным расстоянием в логарифмической шкале
            return min(
                period_medians.keys(),
                key=lambda p: abs(np.log1p(val) - np.log1p(period_medians[p])),
            )

        missing_period_mask = df_out["salary_mean"].notna() & df_out["salary_period"].isna()
        df_out.loc[missing_period_mask, "salary_period"] = df_out.loc[
            missing_period_mask, "salary_mean"
        ].apply(_assign_closest_period)
        df_out["salary_period_imputed"] = missing_period_mask

        logger.info(f"Восстановлен период выплат для {missing_period_mask.sum()} записей.")
        return df_out

    def transform(self, df: pd.DataFrame, col: str = "salary") -> pd.DataFrame:
        """Запускает классификацию формата, парсинг чисел и импутацию периодов для колонки зарплаты."""
        if col not in df.columns:
            return df

        df_out = df.copy()
        df_out["salary_format"] = df_out[col].apply(self.classify_salary_format)
        parsed = df_out[col].apply(self.parse_salary_string).apply(pd.Series)
        df_out = pd.concat([df_out, parsed], axis=1)
        return self.impute_missing_periods(df_out)