import pandas as pd

file_path = 'clients.xlsx'
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names  # листы в файле (предполагаем, что 12 листов)

# Сначала собираем множества ID с каждого листа
id_sets = []
for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    ids = set(df['id'].dropna().unique())
    id_sets.append(ids)

# Находим пересечение - клиенты, которые есть на всех листах
common_ids = set.intersection(*id_sets)

print(f"Общее количество клиентов, присутствующих на всех листах: {len(common_ids)}")

# Теперь собираем данные по этим клиентам для каждого листа
# Предположим, что параметры начинаются с колонки с индексом 1 (после 'id')
# Можно указать конкретные имена параметров, например: params = ['param1', 'param2']
# Для примера возьмем все колонки кроме 'id'

all_data = []

for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    # Фильтруем по общим клиентам
    df = df[df['id'].isin(common_ids)].copy()
    # Добавим колонку с именем месяца (листа), чтобы потом знать порядковый месяц
    df['month'] = sheet
    all_data.append(df)

# Конкатенируем все данные в один DataFrame
full_data = pd.concat(all_data, ignore_index=True)

# Получаем список параметров (все колонки, кроме 'id' и 'month')
params = [col for col in full_data.columns if col not in ['id', 'month']]

# Теперь преобразуем данные в формат "клиент - параметр - месяц" в широкой форме (pivot)
# Для каждого параметра сформируем отдельную таблицу с месяцами по колонкам

time_series_data = {}

for param in params:
    pivot_df = full_data.pivot(index='id', columns='month', values=param)
    # Можно отсортировать столбцы по нужному порядку, если months — это даты или номера
    time_series_data[param] = pivot_df

# Пример: вывести временной ряд для param1
print(time_series_data[params[0]].head())

# Можно сохранить каждый параметр в отдельный Excel лист или CSV
with pd.ExcelWriter('clients_time_series.xlsx') as writer:
    for param, df_ts in time_series_data.items():
        df_ts.to_excel(writer, sheet_name=param[:31])  # листы в Excel ограничены 31 символом

print("Временные ряды клиентов по параметрам сохранены в файл 'clients_time_series.xlsx'")
