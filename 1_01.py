import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# === 1. Подготовка данных ===

# Замените на ваши реальные списки признаков
binary_features = [...]    # список колонок с 0/1
numeric_features = [...]   # список числовых колонок
target = 'y'

# Разделение признаков и целевой переменной
X_bin = df[binary_features]
X_num = df[numeric_features]
y = df[target]

# Масштабируем числовые признаки
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Объединяем бинарные и нормализованные числовые признаки
X_full = pd.concat([X_bin.reset_index(drop=True), 
                    pd.DataFrame(X_num_scaled, columns=numeric_features)], axis=1)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=42, stratify=y)

# === 2. Модель 1: LogisticRegression c class_weight='balanced' ===

model1 = LogisticRegression(solver='saga', penalty='l2', class_weight='balanced', max_iter=1000)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
y_proba1 = model1.predict_proba(X_test)[:, 1]

# === 3. Модель 2: Oversampling + корректировка интерсепта ===

# Собираем датафрейм для oversampling
train_df = X_train.copy()
train_df['y'] = y_train.values

df_majority = train_df[train_df.y == 0]
df_minority = train_df[train_df.y == 1]

# Oversample миноритарного класса до количества majority
df_minority_over = resample(df_minority, 
                            replace=True, 
                            n_samples=len(df_majority), 
                            random_state=42)

df_balanced = pd.concat([df_majority, df_minority_over])
X_train_bal = df_balanced.drop(columns='y')
y_train_bal = df_balanced['y']

# Обучаем модель на сбалансированных данных
model2 = LogisticRegression(solver='saga', penalty='l2', max_iter=1000)
model2.fit(X_train_bal, y_train_bal)

# Корректировка интерсепта по формуле Manski & Lerman
p = y_train.mean()   # реальная доля класса 1 в оригинальных данных
r = 0.5              # в oversampled выборке класс 1 = 50%
b0_raw = model2.intercept_[0]
correction = np.log((p / r) * ((1 - r) / (1 - p)))
model2.intercept_ = np.array([b0_raw + correction])

# Предсказания
y_pred2 = model2.predict(X_test)
y_proba2 = model2.predict_proba(X_test)[:, 1]

# === 4. Оценка моделей ===

def evaluate_model(y_true, y_pred, y_proba, name):
    print(f"\n📊 {name}")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"AUC ROC: {roc_auc_score(y_true, y_proba):.4f}")

evaluate_model(y_test, y_pred1, y_proba1, "Модель 1: class_weight='balanced'")
evaluate_model(y_test, y_pred2, y_proba2, "Модель 2: oversampling + интерсепт")


bins = [-np.inf, -500, -200, -100, -50, -10, 10, 50, 100, 200, 500, np.inf]
labels = [
    'резкое падение',
    'сильное падение',
    'значительное падение',
    'умеренное падение',
    'лёгкое падение',
    'стабильно',
    'лёгкий рост',
    'умеренный рост',
    'значительный рост',
    'сильный рост',
    'резкий рост'
]

# Категоризация
df['delta_turnover_cat'] = pd.cut(df['delta_turnover'], bins=bins, labels=labels)

# Проверим результат
print(df['delta_turnover_cat'].value_counts().sort_index())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='delta_turnover_cat', order=labels)
plt.xticks(rotation=45)
plt.title("Категории изменения оборота (11 групп)")
plt.xlabel("Категория")
plt.ylabel("Количество клиентов")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Список процентных признаков
percent_features = ['delta_turnover', 'change_in_spending', 'margin_change']  # примеры

# Цикл по фичам
for feature in percent_features:
    plt.figure(figsize=(14, 5))

    # Гистограмма
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature], bins=100, kde=True)
    plt.title(f"Распределение: {feature}")
    plt.xlabel("Значение (%)")
    plt.ylabel("Количество")

    # Boxplot (выбросы)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot: {feature}")

    plt.tight_layout()
    plt.show()

    # Статистика
    print(f"\n📊 Статистика по признаку: {feature}")
    print(df[feature].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))
    print("-" * 80)
