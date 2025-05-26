import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Убедимся, что даты в формате datetime
df_all['BASEMINDATE'] = pd.to_datetime(df_all['BASEMINDATE'], errors='coerce')
df_all['report_date'] = pd.to_datetime(df_all['report_date'], errors='coerce')

# Возраст клиента в месяцах
df_all['client_months'] = ((df_all['report_date'] - df_all['BASEMINDATE']).dt.days / 30.44).astype(int)

# Фильтрация от аномалий
df_filtered = df_all[(df_all['client_months'] >= 0) & df_all['target'].notna()]

# Группировка по возрасту в месяцах
month_stats = df_filtered.groupby('client_months')['target'].agg(['mean', 'count']).reset_index()
month_stats.rename(columns={'mean': 'leave_rate'}, inplace=True)

# Хи-квадрат тест
contingency = pd.crosstab(df_filtered['client_months'], df_filtered['target'])
chi2, p, dof, expected = chi2_contingency(contingency)

# График
plt.figure(figsize=(14,6))
sns.lineplot(x='client_months', y='leave_rate', data=month_stats, marker='o', color='darkorange')
plt.title('Зависимость ухода от времени пребывания клиента (в месяцах)')
plt.xlabel('Сколько месяцев клиент с нами')
plt.ylabel('Доля ушедших')
plt.grid(True)

# Добавим p-value
plt.annotate(f'p-value = {p:.4f}', xy=(0.01, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round", fc="w"))

plt.tight_layout()
plt.show()
