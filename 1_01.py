import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Загрузка Excel-файла
base = "your_file.xlsx"  # ← Укажи путь к файлу
sheet_names = pd.ExcelFile(base).sheet_names

# Инициализация списков
current_clients, lost_clients, new_clients = [], [], []
all_seen_clients = set()

# Сбор данных по листам
for i, sheet in enumerate(tqdm(sheet_names, desc="Обработка месяцев")):
    df = pd.read_excel(base, sheet_name=sheet)

    clients = set(df['CLIENTBASENUMBER'].unique())
    cur_clients = set(df.loc[df['pr'] == 0, 'CLIENTBASENUMBER'].unique())
    lost = set(df.loc[df['pr'] == 1, 'CLIENTBASENUMBER'].unique())

    current_clients.append(len(cur_clients))
    lost_clients.append(len(lost))

    if i == 0:
        new_clients.append(np.nan)
    else:
        new_clients.append(len(cur_clients - all_seen_clients))

    all_seen_clients.update(clients)

# Расчёт оттока клиентов в процентах
churn_rate = [np.nan]
for i in range(1, len(current_clients)):
    total_prev_month = current_clients[i - 1]
    lost = lost_clients[i]
    if total_prev_month > 0:
        churn_rate.append(lost / total_prev_month * 100)
    else:
        churn_rate.append(np.nan)

# Создание общего DataFrame для графиков
df_plot = pd.DataFrame({
    'Месяц': sheet_names,
    'Текущие клиенты': current_clients,
    'Ушедшие клиенты': lost_clients,
    'Новые клиенты': new_clients,
    'Отток (%)': churn_rate
})

# Настройка стиля графиков
sns.set(style="whitegrid")
plt.figure(figsize=(16, 14))

# График 1
plt.subplot(4, 1, 1)
sns.lineplot(data=df_plot, x='Месяц', y='Текущие клиенты', marker='o', color='blue')
plt.title('Текущие клиенты')

# График 2
plt.subplot(4, 1, 2)
sns.lineplot(data=df_plot, x='Месяц', y='Ушедшие клиенты', marker='o', color='red')
plt.title('Ушедшие клиенты')

# График 3
plt.subplot(4, 1, 3)
sns.lineplot(data=df_plot, x='Месяц', y='Новые клиенты', marker='o', color='green')
plt.title('Новые клиенты')

# График 4
plt.subplot(4, 1, 4)
sns.lineplot(data=df_plot, x='Месяц', y='Отток (%)', marker='o', color='orange')
plt.title('Отток клиентов в процентах')
plt.ylabel('%')

plt.tight_layout()
plt.show()
