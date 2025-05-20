import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sheet_names = ['202301', '202302', '202303', '202304', '202305', '202306',
               '202307', '202308', '202309', '202310', '202311', '202312']

xls = 'clients.xlsx'

total_clients = []
lost_clients = []
new_clients = []

all_seen_clients = set()

for i, sheet in enumerate(tqdm(sheet_names, desc="Обработка месяцев")):
    df = pd.read_excel(xls, sheet_name=sheet)
    clients = set(df['CLIENTBASENUMBER'].unique())
    total_clients.append(len(clients))
    lost_clients.append(df.loc[df['pr'] == True, 'CLIENTBASENUMBER'].nunique())
    if i == 0:
        new_clients.append(np.nan)
    else:
        new = clients - all_seen_clients
        new_clients.append(len(new))
    all_seen_clients.update(clients)

def add_bar_labels(ax, values):
    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(i, v + max(values) * 0.01, str(int(v)), ha='center', va='bottom', fontsize=9)

fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)

axes[0].bar(sheet_names, total_clients, color='steelblue')
axes[0].set_title('Общее число клиентов')
axes[0].set_xlabel('Месяц')
axes[0].set_ylabel('Уникальные клиенты')
axes[0].tick_params(axis='x', rotation=45)
add_bar_labels(axes[0], total_clients)

axes[1].bar(sheet_names, lost_clients, color='indianred')
axes[1].set_title('Ушедшие клиенты')
axes[1].set_xlabel('Месяц')
axes[1].tick_params(axis='x', rotation=45)
add_bar_labels(axes[1], lost_clients)

axes[2].bar(sheet_names, new_clients, color='seagreen')
axes[2].set_title('Пришедшие клиенты')
axes[2].set_xlabel('Месяц')
axes[2].tick_params(axis='x', rotation=45)
add_bar_labels(axes[2], new_clients)

plt.tight_layout()
plt.show()
