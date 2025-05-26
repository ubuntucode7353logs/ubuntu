import pandas as pd
from tqdm import tqdm

# Словарь для хранения пропущенных значений
missing_dict = {}

# Проходим по всем листам
for sheet_name in tqdm(sheet_names, desc="Проверка пропущенных значений:"):
    sheet = base.parse(sheet_name=sheet_name)
    missing = sheet.isnull().sum()
    missing = missing[missing > 0]
    for col, count in missing.items():
        if col not in missing_dict:
            missing_dict[col] = {}
        missing_dict[col][sheet_name] = count

# Преобразуем в DataFrame
missing_matrix = pd.DataFrame(missing_dict).T.fillna(0).astype(int)

# Упорядочим столбцы по времени (если они в формате 'YYYYMM')
missing_matrix = missing_matrix[sorted(missing_matrix.columns)]

# Выводим таблицу
missing_matrix

