import logging
from pathlib import Path
import pandas as pd

from src.cleaner import DataQualityCleaner
from src.location_parser import LocationParser
from src.salary_parser import SalaryParser
from src.visualizer import DataQualityVisualizer

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path,
    output_path: Path,
    reports_dir: Path = Path("reports"),
    generate_plots: bool = True,
) -> pd.DataFrame:
    """Собирает весь пайплайн: аудит колонок, очистка, парсинг локаций/зарплат и генерация графиков."""
    logger.info(f"Загружаем исходный датасет из {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Исходный размер таблицы: {df.shape}")

    cleaner = DataQualityCleaner()
    df = cleaner.normalize_missing_values(df)

    # Проверка типов, пропусков и пересечений значений между колонками
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_df = cleaner.generate_column_audit_table(df)
    audit_df.to_csv(reports_dir / "column_quality_audit.csv", index=False)

    intersections_df = cleaner.find_column_intersections(df)
    intersections_df.to_csv(reports_dir / "column_value_intersections.csv", index=False)

    # дропаем константные колонки, чиним сдвиги organization <-> location и убираем дубликаты
    df, _ = cleaner.drop_constant_columns(df)
    df = cleaner.fix_organization_location_swaps(df)
    df = cleaner.remove_duplicates(df)

    # Вытаскиваем структурированные фичи по локациям и зарплатам
    df = LocationParser().transform(df, col="location")
    df = SalaryParser().transform(df, col="salary")

    # Строим и сохраняем графики для отчета
    if generate_plots:
        visualizer = DataQualityVisualizer(output_dir=reports_dir / "figures")
        visualizer.generate_all_reports(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Очищенный датасет сохранен в {output_path} (Итоговый размер: {df.shape})")
    return df