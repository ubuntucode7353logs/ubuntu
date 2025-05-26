from scipy.stats import mannwhitneyu

# Разобьём возраст на группы (например, кварталы)
df_all['age_bin'] = pd.cut(df_all['client_months'], bins=np.arange(0, df_all['client_months'].max()+3, 3), right=False)

# Среднее pr по бинам
bin_stats = df_all.groupby('age_bin')['pr'].mean()

print(bin_stats)

# Тест Манна-Уитни между двумя группами (например, младшие и старшие)
group1 = df_all[df_all['client_months'] <= 12]['pr'].dropna()
group2 = df_all[df_all['client_months'] > 12]['pr'].dropna()

stat, p = mannwhitneyu(group1, group2)
print(f'Mann-Whitney U тест: stat={stat}, p-value={p}')
