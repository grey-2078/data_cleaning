import re
from typing import List, Set

# Текстовые плейсхолдеры и мусорные значения, которые сразу переводим в NaN
MISSING_VALUE_PLACEHOLDERS: Set[str] = {
    "n/a", "na", "none", "null", "nan", "-", "--", "unknown", "unspecified", "not specified"
}

# Регулярка для строк, где остались только пробелы или табы
WHITESPACE_REGEX = re.compile(r"^\s*$")

# Регулярки для разбора локации (индекс, штат, стандартный формат "City, ST Zip")
ZIP_CODE_REGEX = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
STATE_CODE_REGEX = re.compile(r"\b([A-Z]{2})\b")
CITY_STATE_ZIP_REGEX = re.compile(
    r"^(?P<city>[A-Za-z\s\.\-']+?),\s*(?P<state>[A-Z]{2})(?:\s+(?P<zip>\d{5}(?:-\d{4})?))?$"
)

# Регулярки и ключевые слова для парсинга зарплат и периода выплат
SALARY_NUMBER_REGEX = re.compile(r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*([kK])?")
SALARY_PERIOD_KEYWORDS = {
    "year": ["year", "yr", "annual", "annually", "pa", "p.a."],
    "hour": ["hour", "hr", "hourly"],
    "month": ["month", "mo", "monthly"],
    "week": ["week", "wk", "weekly"],
    "day": ["day", "daily"],
}

# Поля для поиска семантических дубликатов (без учета uniq_id и технических ссылок)
SEMANTIC_DUPLICATE_SUBSET: List[str] = [
    "job_title",
    "job_description",
    "location",
    "organization",
    "salary",
]