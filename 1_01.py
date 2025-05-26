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
