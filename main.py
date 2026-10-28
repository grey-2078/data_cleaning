import argparse
import logging
from pathlib import Path
from src.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Пайплайн очистки и обеспечения качества данных (Monster.com)")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/monster_com_job_sample.csv"),
        help="Путь к исходному CSV-файлу",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/monster_jobs_cleaned.csv"),
        help="Куда сохранить очищенный датасет",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Папка для сохранения таблиц аудита и графиков",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не генерировать графики matplotlib/seaborn",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        reports_dir=args.reports_dir,
        generate_plots=not args.no_plots,
    )