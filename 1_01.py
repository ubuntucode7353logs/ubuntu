import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
all_dfs = []
for sheet_name in xls.sheet_names:
    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Добавляем колонку с датой отчёта
    df_sheet['report_date'] = pd.to_datetime(sheet_name, format='%Y%m')
    
    # Преобразуем даты регистрации в datetime
    df_sheet['REGDATE'] = pd.to_datetime(df_sheet['REGDATE'], errors='coerce')
    df_sheet['BASEMINDATE'] = pd.to_datetime(df_sheet['BASEMINDATE'], errors='coerce')
    
    # Считаем возраст в днях на момент отчёта
    df_sheet['age_days_REGDATE'] = (df_sheet['report_date'] - df_sheet['REGDATE']).dt.days
    df_sheet['age_days_BASEMINDATE'] = (df_sheet['report_date'] - df_sheet['BASEMINDATE']).dt.days
    
    df_sheet = df_sheet.drop(columns=['REGDATE', 'BASEMINDATE'])
    df_sheet['target'] = df_sheet['target'].fillna(0)
    
    all_dfs.append(df_sheet)

df_all = pd.concat(all_dfs, ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.boxplot(x='target', y='age_days_REGDATE', data=df_all)
plt.xticks([0,1], ['Остался', 'Ушёл'])
plt.xlabel('Статус клиента')
plt.ylabel('Возраст клиента (дней)')
plt.title('Распределение возраста по статусу клиента')
plt.show()

from scipy.stats import ttest_ind

group0 = df_all[df_all['target'] == 0]['age_days_REGDATE'].dropna()
group1 = df_all[df_all['target'] == 1]['age_days_REGDATE'].dropna()

t_stat, p_val = ttest_ind(group0, group1, equal_var=False)  # Welch's t-test

print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")

def phi_coefficient(contingency):
    """Вычислить коэффициент Фи из таблицы сопряжённости 2x2"""
    # таблица 2x2: [[a, b], [c, d]]
    a = contingency.iloc[0,0]
    b = contingency.iloc[0,1]
    c = contingency.iloc[1,0]
    d = contingency.iloc[1,1]
    numerator = (a*d - b*c)
    denominator = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    if denominator == 0:
        return np.nan
    return numerator / denominator

def odds_ratio(contingency):
    """Вычислить Odds Ratio из таблицы сопряжённости 2x2"""
    a = contingency.iloc[0,0]
    b = contingency.iloc[0,1]
    c = contingency.iloc[1,0]
    d = contingency.iloc[1,1]
    # Для избежания деления на 0 добавим 0.5 к ячейкам (Haldane-Anscombe correction)
    a += 0.5
    b += 0.5
    c += 0.5
    d += 0.5
    return (a*d) / (b*c)

def analyze_bool_vars(df, target_col, bool_vars):
    results = []
    for var in bool_vars:
        print(f"--- Анализ переменной: {var} ---")
        contingency = pd.crosstab(df[var], df[target_col])
        print("Таблица сопряжённости:")
        print(contingency)
        
        # Хи-квадрат тест
        chi2, p, dof, ex = chi2_contingency(contingency)
        print(f"Chi2: {chi2:.3f}, p-value: {p:.4f}")
        
        # Коэффициент Фи
        if contingency.shape == (2, 2):
            phi = phi_coefficient(contingency)
            oratio = odds_ratio(contingency)
            print(f"Phi coefficient: {phi:.3f}")
            print(f"Odds Ratio: {oratio:.3f}")
        else:
            phi = np.nan
            oratio = np.nan
            print("Коэффициент Фи и Odds Ratio рассчитываются только для таблиц 2x2")
        
        # Визуализация
        proportions = contingency.div(contingency.sum(axis=1), axis=0)
        proportions.plot(kind='bar', stacked=True)
        plt.title(f"Доли таргета по значениям {var}")
        plt.xlabel(var)
        plt.ylabel("Доля")
        plt.legend(title=target_col)
        plt.show()
        
        results.append({
            'variable': var,
            'chi2': chi2,
            'p_value': p,
            'phi_coefficient': phi,
            'odds_ratio': oratio
        })
    return pd.DataFrame(results)

# Пример вызова функции
# df — датафрейм, target — имя колонки с таргетом (0/1), bool_vars — список булевых переменных
# results_df = analyze_bool_vars(df, 'target', bool_vars)
# print(results_df)


