import pandas as pd
import numpy as np

# Пример: датафрейм с cts1–cts3
# df = pd.read_csv("your_data.csv")

# 1. Вычисляем среднюю сумму поступлений за 3 месяца
df['cts_avg'] = df[['cts1', 'cts2', 'cts3']].mean(axis=1)

# 2. Определяем пороги и метки
thresholds = [0, 1_000, 10_000, 100_000, 1_000_000, np.inf]
labels = [0, 1, 2, 3, 4]  # Можно заменить на текстовые: ['сотни', 'тысячи', ...]

# 3. Присваиваем кластер на основе диапазонов
df['cts_cluster'] = pd.cut(df['cts_avg'], bins=thresholds, labels=labels, right=False).astype(int)
