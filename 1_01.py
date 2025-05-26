from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

target = 'pr'
features = df.columns.drop(target)

# 1. Пропуски
df = df.fillna(0)

# 2. Разделение признаков
X = df[features]
y = df[target]

# 3. Масштабирование только числовых признаков
numeric_cols = X.select_dtypes(include=['float', 'int']).columns
bool_cols = X.select_dtypes(include='bool').columns

scaler = StandardScaler()
X_scaled_numeric = scaler.fit_transform(X[numeric_cols])

# 4. Объединение: числовые (масштабированные) + булевые (прямо)
X_final = pd.DataFrame(X_scaled_numeric, columns=numeric_cols)
X_final[bool_cols] = X[bool_cols].astype(int).values

# 5. Модель
model = LogisticRegression(max_iter=1000)
model.fit(X_final, y)

# 6. Коэффициенты
import pandas as pd
coeffs = pd.Series(model.coef_[0], index=X_final.columns).sort_values(key=abs, ascending=False)
print(coeffs)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Параметры
y_limits = [(28000, 37000), (600, 3000), (700, 2500), (0, 10)]
titles = ['Текущие клиенты', 'Потерянные клиенты', 'Новые клиенты', 'Отток клиентов в процентах']
data_list = [current_clients, lost_clients, new_clients, churn_rate]
colors = ['skyblue', 'salmon', 'mediumseagreen', 'orange']

# Создание подграфиков
fig, axs = plt.subplots(1, 4, figsize=(30, 6), facecolor='white')

# Построение графиков
for i, (ax, data, ylim, title, color) in enumerate(zip(axs, data_list, y_limits, titles, colors)):
    df = pd.DataFrame({'Месяц': sheet_names, 'Значение': data})
    sns.barplot(data=df, x='Месяц', y='Значение', ax=ax, color=color)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=45)

    # Добавление подписей к столбцам
    for index, value in enumerate(data):
        ax.text(index, value + (ylim[1] - ylim[0]) * 0.02, str(value), ha='center', va='bottom')

plt.tight_layout()
plt.show()
