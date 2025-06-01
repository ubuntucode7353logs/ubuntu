df = pd.read_excel("file.xlsx", sheet_name="202301", parse_dates=['REGDATE', 'BASEMINDATE'])

df['REGDATE'] = pd.to_datetime(df['REGDATE'], errors='coerce', dayfirst=True)

df_all['BASEMINDATE'] = pd.to_datetime(df_all['BASEMINDATE'], errors='coerce')
df_all['report_date'] = pd.to_datetime(df_all['report_date'], errors='coerce')

# Считаем возраст клиента в месяцах
df_all['client_months'] = ((df_all['report_date'] - df_all['BASEMINDATE']).dt.days / 30.44).astype(int)

# Отфильтруем валидные строки
df_filtered = df_all[(df_all['client_months'] >= 0) & df_all['target'].notna()]

# Группировка по кварталам (каждые 3 месяца)
df_filtered['months_bin'] = pd.cut(
    df_filtered['client_months'],
    bins=np.arange(0, df_filtered['client_months'].max() + 3, 3),
    right=False,
    include_lowest=True
)

# Статистика по квартальным биннам
bin_stats = df_filtered.groupby('months_bin')['target'].agg(['mean', 'count']).reset_index()
bin_stats.rename(columns={'mean': 'leave_rate'}, inplace=True)

# Хи-квадрат тест
contingency = pd.crosstab(df_filtered['months_bin'], df_filtered['target'])
chi2, p, dof, expected = chi2_contingency(contingency)

# Гистограмма
plt.figure(figsize=(14,6))
sns.barplot(x=bin_stats['months_bin'].astype(str), y='leave_rate', data=bin_stats, color='steelblue')
plt.xticks(rotation=45)
plt.title('Уход клиентов в зависимости от срока пребывания (по кварталам)')
plt.xlabel('Сколько месяцев клиент с нами (квартальные интервалы)')
plt.ylabel('Доля ушедших')
plt.grid(axis='y')

# Добавим p-value
plt.annotate(f'p-value = {p:.4f}', xy=(0.01, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round", fc="w"))

plt.tight_layout()
plt.show()
df_all['report_date'] = pd.to_datetime(df_all['report_date'], errors='coerce')

# Считаем возраст клиента в месяцах
df_all['client_months'] = ((df_all['report_date'] - df_all['BASEMINDATE']).dt.days / 30.44).astype(int)

# Отфильтруем валидные строки
df_filtered = df_all[(df_all['client_months'] >= 0) & df_all['target'].notna()]

# Группировка по кварталам (каждые 3 месяца)
df_filtered['months_bin'] = pd.cut(
    df_filtered['client_months'],
    bins=np.arange(0, df_filtered['client_months'].max() + 3, 3),
    right=False,
    include_lowest=True
)

# Статистика по квартальным биннам
bin_stats = df_filtered.groupby('months_bin')['target'].agg(['mean', 'count']).reset_index()
bin_stats.rename(columns={'mean': 'leave_rate'}, inplace=True)

# Хи-квадрат тест
contingency = pd.crosstab(df_filtered['months_bin'], df_filtered['target'])
chi2, p, dof, expected = chi2_contingency(contingency)

# Гистограмма
plt.figure(figsize=(14,6))
sns.barplot(x=bin_stats['months_bin'].astype(str), y='leave_rate', data=bin_stats, color='steelblue')
plt.xticks(rotation=45)
plt.title('Уход клиентов в зависимости от срока пребывания (по кварталам)')
plt.xlabel('Сколько месяцев клиент с нами (квартальные интервалы)')
plt.ylabel('Доля ушедших')
plt.grid(axis='y')

# Добавим p-value
plt.annotate(f'p-value = {p:.4f}', xy=(0.01, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round", fc="w"))

plt.tight_layout()
plt.show()f_all['BASEMINDATE'] = pd.to_datetime(df_all['BASEMINDATE'], errors='coerce')
df_all['report_date'] = pd.to_datetime(df_all['report_date'], errors='coerce')

# Считаем возраст клиента в месяцах
df_all['client_months'] = ((df_all['report_date'] - df_all['BASEMINDATE']).dt.days / 30.44).astype(int)

# Отфильтруем валидные строки
df_filtered = df_all[(df_all['client_months'] >= 0) & df_all['target'].notna()]

# Группировка по кварталам (каждые 3 месяца)
df_filtered['months_bin'] = pd.cut(
    df_filtered['client_months'],
    bins=np.arange(0, df_filtered['client_months'].max() + 3, 3),
    right=False,
    include_lowest=True
)

# Статистика по квартальным биннам
bin_stats = df_filtered.groupby('months_bin')['target'].agg(['mean', 'count']).reset_index()
bin_stats.rename(columns={'mean': 'leave_rate'}, inplace=True)

# Хи-квадрат тест
contingency = pd.crosstab(df_filtered['months_bin'], df_filtered['target'])
chi2, p, dof, expected = chi2_contingency(contingency)

# Гистограмма
plt.figure(figsize=(14,6))
sns.barplot(x=bin_stats['months_bin'].astype(str), y='leave_rate', data=bin_stats, color='steelblue')
plt.xticks(rotation=45)
plt.title('Уход клиентов в зависимости от срока пребывания (по кварталам)')
plt.xlabel('Сколько месяцев клиент с нами (квартальные интервалы)')
plt.ylabel('Доля ушедших')
plt.grid(axis='y')

# Добавим p-value
plt.annotate(f'p-value = {p:.4f}', xy=(0.01, 0.95), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round", fc="w"))

plt.tight_layout()
plt.show()
