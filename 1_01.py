import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка Excel с листами
xls = pd.ExcelFile('clients.xlsx')
months = ['202301', '202302', '202303', '202304', '202305', '202306', '202307', '202308', '202309', '202310', '202311', '202312']
data = {month: xls.parse(month) for month in months}

total_clients = []
lost_clients = []
new_clients = []

all_seen_clients = set()

for month in months:
    df = data[month]
    clients = set(df['CLIENTBASENUMBER'].unique())  # уникальные клиенты в месяце
    
    total_clients.append(len(clients))
    lost_clients.append(df.loc[df['pr'] == True, 'CLIENTBASENUMBER'].nunique())  # ушедшие по уникальным
    
    new = clients - all_seen_clients  # новые клиенты — не встречались ранее
    new_clients.append(len(new))
    
    all_seen_clients.update(clients)

# Построение графика
x = np.arange(len(months))
width = 0.25

fig, ax = plt.subplots(figsize=(15,6))
ax.bar(x - width, total_clients, width, label='Общее число клиентов')
ax.bar(x, lost_clients, width, label='Ушедшие клиенты')
ax.bar(x + width, new_clients, width, label='Пришедшие клиенты')

ax.set_xticks(x)
ax.set_xticklabels(months, rotation=45)
ax.set_xlabel('Месяц')
ax.set_ylabel('Количество уникальных клиентов')
ax.set_title('Уникальные клиенты по месяцам: общее, ушедшие, пришедшие')
ax.legend()

plt.tight_layout()
plt.show()
