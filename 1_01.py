from scipy.stats import mannwhitneyu

age_left = df_all[df_all['pr'] == 1]['age_days_REGDATE']
age_stayed = df_all[df_all['pr'] == 0]['age_days_REGDATE']

stat, p = mannwhitneyu(age_left, age_stayed, alternative='two-sided')
print(f'Mann–Whitney U test: stat = {stat:.2f}, p-value = {p:.4f}')
