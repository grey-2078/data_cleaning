import logging
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Настраиваем базовый стиль графиков
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120


class DataQualityVisualizer:
    """Строит и сохраняет графики для отчета по качеству данных (пропуски, форматы локаций и зарплат)."""

    def __init__(self, output_dir: Path = Path("reports/figures")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_or_show(self, fig: plt.Figure, filename: Optional[str]) -> None:
        fig.tight_layout()
        if filename:
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            logger.info(f"График сохранен: {out_path}")
        plt.close(fig)

    def plot_missing_values(
        self, df: pd.DataFrame, filename: Optional[str] = "missing_values.png"
    ) -> None:
        """Строит распределение доли пропусков по колонкам (в процентах)."""
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]

        if missing_pct.empty:
            logger.info("Пропусков нет — график пропускаем.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=missing_pct.values,
            y=missing_pct.index,
            hue=missing_pct.index,
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title("Missing Values Percentage by Column", fontsize=14, fontweight="bold")
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_ylabel("Column Name")
        for i, v in enumerate(missing_pct.values):
            ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, max(missing_pct.values) * 1.12)
        self._save_or_show(fig, filename)

    def plot_location_format_distribution(
        self, df: pd.DataFrame, col: str = "location_format", filename: Optional[str] = "location_formats.png"
    ) -> None:
        """Визуализирует распределение форматов записи локации в датасете."""
        if col not in df.columns:
            return

        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x=counts.values,
            y=counts.index,
            hue=counts.index,
            palette="Blues_r",
            legend=False,
            ax=ax,
        )
        ax.set_title("Distribution of Raw Location Formats", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Job Postings")
        ax.set_ylabel("Location Format Category")
        for i, v in enumerate(counts.values):
            ax.text(v + max(counts.values) * 0.01, i, f"{v:,}", va="center", fontsize=9)
        ax.set_xlim(0, max(counts.values) * 1.15)
        self._save_or_show(fig, filename)

    def plot_salary_format_distribution(
        self, df: pd.DataFrame, col: str = "salary_format", filename: Optional[str] = "salary_formats.png"
    ) -> None:
        """Строит распределение форматов зарплат (без учета пустых значений)."""
        if col not in df.columns:
            return

        counts = df[df[col] != "Missing"][col].value_counts()
        if counts.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x=counts.values,
            y=counts.index,
            hue=counts.index,
            palette="magma",
            legend=False,
            ax=ax,
        )
        ax.set_title("Distribution of Non-Empty Salary Formats", fontsize=14, fontweight="bold")
        ax.set_xlabel("Count")
        ax.set_ylabel("Salary Format Category")
        for i, v in enumerate(counts.values):
            ax.text(v + max(counts.values) * 0.01, i, f"{v:,}", va="center", fontsize=9)
        ax.set_xlim(0, max(counts.values) * 1.15)
        self._save_or_show(fig, filename)

    def plot_salary_stats_by_period(
        self, df: pd.DataFrame, filename: Optional[str] = "salary_by_period.png"
    ) -> None:
        """Сравнивает среднюю и медианную зарплату в разрезе периодов выплат (лог-шкала)."""
        if "salary_mean" not in df.columns or "salary_period" not in df.columns:
            return

        stats = (
            df.dropna(subset=["salary_mean", "salary_period"])
            .groupby("salary_period")["salary_mean"]
            .agg(["count", "median", "mean"])
            .reset_index()
        )
        if stats.empty:
            return

        melted = stats.melt(
            id_vars=["salary_period", "count"],
            value_vars=["median", "mean"],
            var_name="Metric",
            value_name="USD",
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=melted,
            x="salary_period",
            y="USD",
            hue="Metric",
            palette="Set2",
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title("Mean vs. Median Salary by Payment Period (Log Scale)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Payment Period")
        ax.set_ylabel("Salary in USD (Log Scale)")
        self._save_or_show(fig, filename)

    def generate_all_reports(self, df: pd.DataFrame) -> None:
        """Генерирует сразу все графики и складывает их в папку reports/figures/."""
        self.plot_missing_values(df)
        self.plot_location_format_distribution(df)
        self.plot_salary_format_distribution(df)
        self.plot_salary_stats_by_period(df)