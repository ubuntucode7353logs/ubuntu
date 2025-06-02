df_all = pd.concat([pd.DataFrame(lst) for lst in data_per_sheet.values()])

# Группировка и вычисление среднего
df_avg = df_all.groupby('feature', as_index=False)['percent_nonzero'].mean()

# Построение графика
plt.figure(figsize=(8, 5))
sns.barplot(data=df_avg, x='feature', y='percent_nonzero', hue='percent_nonzero', palette='Oranges')

# Добавление подписей
for i, row in df_avg.iterrows():
    plt.text(i, row['percent_nonzero'] + 0.5, f"{row['percent_nonzero']:.1f}%",
             ha='center', va='bottom', fontsize=8)

plt.title('Средняя заполненность признаков (в процентах)')
plt.ylabel('Процент заполненности')
plt.xlabel('')
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
