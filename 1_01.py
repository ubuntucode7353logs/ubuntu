import statsmodels.api as sm

# Удалим строки с NaN (если такие есть в возрасте или таргете)
df_reg = df_all[['pr', 'age_days_REGDATE']].dropna()

# Зависимая переменная (таргет)
y = df_reg['pr']

# Независимая переменная (возраст) + константа
X = sm.add_constant(df_reg['age_days_REGDATE'])

# Логистическая регрессия
model = sm.Logit(y, X).fit()

# Результаты
print(model.summary())
