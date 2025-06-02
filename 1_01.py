import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Твои данные (пример)
data = [
    [{'feature': 'dcts1', 'nonzero_count': 27399, 'percent_nonzero': 86.8376},
     {'feature': 'dcts2', 'nonzero_count': 28520, 'percent_nonzero': 90.3905},
     {'feature': 'dcts3', 'nonzero_count': 29133, 'percent_nonzero': 92.3333},
     {'feature': 'dctq1', 'nonzero_count': 24656, 'percent_nonzero': 78.1440},
     {'feature': 'dctq2', 'nonzero_count': 25634, 'percent_nonzero': 81.2437},
     {'feature': 'dctq3', 'nonzero_count': 26064, 'percent_nonzero': 82.6065},
     {'feature': 'ddts1', 'nonzero_count': 28773, 'percent_nonzero': 91.1923},
     {'feature': 'ddts2', 'nonzero_count': 30163, 'percent_nonzero': 95.5977},
     {'feature': 'ddts3', 'nonzero_count': 30858, 'percent_nonzero': 97.8005},
     {'feature': 'ddtq1', 'nonzero_count': 27704}]  # Заметим: нет percent_nonzero!
    # Добавь другие месяцы по аналогии, если нужно
]

# Объединяем всё в один DataFrame
df_all = pd.concat([pd.DataFrame(sheet) for sheet in data], ignore_index=True)

# Убираем строки без percent_nonzero
df_all = df_all.dropna(subset=['percent_nonzero'])

# Группируем по признаку и считаем среднее
df_avg = df_all.groupby('feature', as_index=False)['percent_nonzero'].mean()

# Строим график
plt.figure(figsize=(10, 6))
sns.barplot(data=df_avg, x='feature', y='percent_nonzero', palette='crest')

# Добавляем подписи над столбцами
for i, row in df_avg.iterrows():
    plt.text(i, row['percent_nonzero'] + 0.5, f"{row['percent_nonzero']:.1f}%",
             ha='center', va='bottom', fontsize=8)

plt.title('Средняя заполненность признаков (в процентах)', fontsize=14)
plt.ylabel('Процент заполненности')
plt.xlabel('')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
