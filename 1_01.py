import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

coefs_list, roc_aucs, corrected_intercepts, corrected_probs_list = [], [], [], []

# ⚙️ Указать реальную долю оттока в популяции:
real_event_rate = 0.04  # например, отток ≈ 4%

for sheet_name in tqdm(sheet_names, desc="Обработка месяцев"):
    df_sheet = base.parse(sheet_name=sheet_name)
    df_sheet['pr'] = df_sheet['pr'].fillna(0)

    X_percent, X_bool = df_sheet[percent_features], df_sheet[bool_features]
    y = df_sheet['pr']

    percent_scaler = StandardScaler()
    X_percent_scaled = percent_scaler.fit_transform(X_percent)
    X_all = np.hstack([X_percent_scaled, X_bool.values])
    feature_names = percent_features + bool_features

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)

    # 🔢 Считаем долю положительного класса в трейне (в oversampled виде)
    sample_event_rate = y_train.mean()

    # 🔁 Коррекция интерсепта по формуле из whitepaper
    intercept_adj = np.log((real_event_rate * (1 - sample_event_rate)) /
                           (sample_event_rate * (1 - real_event_rate)))
    corrected_intercept = model.intercept_[0] + intercept_adj
    corrected_intercepts.append(corrected_intercept)

    # 🧮 Считаем откорректированные вероятности вручную
    logits = X_test @ model.coef_.T + corrected_intercept
    corrected_proba = 1 / (1 + np.exp(-logits))
    corrected_probs_list.append(corrected_proba)

    # 📈 Метрики (можно оставить и y_proba тоже для сравнения)
    roc_auc = roc_auc_score(y_test, corrected_proba)
    class_report = classification_report(y_test, corrected_proba > 0.5, digits=3)
    conf_matrix = confusion_matrix(y_test, corrected_proba > 0.5)

    print(f"\n📅 Месяц: {sheet_name}")
    print(f"ROC AUC (corrected): {roc_auc:.3f}")
    print("Классификационный отчёт:\n", class_report)
    print("Матрица ошибок:\n", conf_matrix)

    # 💾 Коэффициенты
    coefs_list.append(pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    }))
    roc_aucs.append(roc_auc)

# 📊 Визуализация
fig, axes = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
axes = axes.flatten()

for i, sheet_name in tqdm(enumerate(sheet_names), desc="Визуализация месяцев"):
    df_top = coefs_list[i].sort_values(by='coefficient', key=abs, ascending=False).head(15)
    sns.barplot(data=df_top, x='coefficient', y='feature', hue='feature', dodge=False,
                palette='coolwarm', ax=axes[i])
    axes[i].set_title(f'{sheet_name}\nROC AUC: {roc_aucs[i]:.3f}')
    axes[i].grid(True)
