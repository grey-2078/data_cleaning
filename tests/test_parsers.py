import pandas as pd
from src.cleaner import DataQualityCleaner
from src.location_parser import LocationParser
from src.salary_parser import SalaryParser


def test_extract_location_parts_standard():
    # Проверяем разбор стандартного адреса "City, ST Zip"
    res = LocationParser.extract_location_parts("Madison, WI 53702")
    assert res["city"] == "Madison"
    assert res["state"] == "WI"
    assert res["zip_code"] == "53702"


def test_salary_parsing_and_imputation():
    # Проверяем парсинг вилок зарплат и восстановление пропущенного периода по медианам
    df = pd.DataFrame({
        "salary": [
            "$50,000 - $70,000 a year",
            "$20.00 - $25.00 / hour",
            "$60,000 - $65,000",  # Период не указан — должен восстановиться как 'year'
            "$22.50",             # Период не указан — должен восстановиться как 'hour'
        ]
    })
    out = SalaryParser().transform(df)
    assert out.loc[0, "salary_mean"] == 60000.0
    assert out.loc[2, "salary_period"] == "year"
    assert out.loc[3, "salary_period"] == "hour"
    assert bool(out.loc[2, "salary_period_imputed"]) is True


def test_organization_location_swap_fix():
    # Проверяем починку строк, где адрес уехал в колонку organization
    df = pd.DataFrame({
        "organization": ["Dallas, TX 75201", "Acme Corp"],
        "location": ["Some noisy job description text here...", "Austin, TX 73301"],
    })
    fixed = DataQualityCleaner.fix_organization_location_swaps(df)
    assert fixed.loc[0, "location"] == "Dallas, TX 75201"
    assert pd.isna(fixed.loc[0, "organization"])