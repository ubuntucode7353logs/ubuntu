import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

# ==== 1. Генерация данных ====
n_clients = 10
n_months = 12
client_ids = [f'C{str(i).zfill(3)}' for i in range(n_clients)]

data = []

for client in client_ids:
    base = np.random.uniform(1000, 3000)
    series = base + np.random.normal(0, 100, size=n_months)

    if np.random.rand() < 0.5:
        drop_point = np.random.randint(6, 10)
        series[drop_point:] -= np.random.uniform(400, 1000)
        churn = 1
    else:
        churn = 0

    for month in range(n_months):
        data.append({
            'client_id': client,
            'month': month + 1,
            'amount': round(series[month], 2),
            'churn': churn if month == n_months else None
        })

df = pd.DataFrame(data)

# ==== 2. Функция Z-score аномалий ====
def rolling_zscore(series, window=4, threshold=2.5):
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    z = (series - rolling_mean) / (rolling_std + 1e-6)
    return z.abs() > threshold

# ==== 3. Применение на всех клиентах ====
results = []

for client_id, group in df.groupby("client_id"):
    group = group.sort_values("month")
    amount_series = group["amount"].reset_index(drop=True)
    z_flags = rolling_zscore(amount_series, window=4, threshold=2.5)

    # Аномалия в последних 3 месяцах?
    anomaly_in_tail = z_flags[-3:].any()
    churn = group["churn"].dropna().values[0]
    
    results.append({
        "client_id": client_id,
        "churn": churn,
        "anomaly_detected": int(anomaly_in_tail)
    })

res_df = pd.DataFrame(results)

# ==== 4. Оценка качества ====
y_true = res_df["churn"]
y_pred = res_df["anomaly_detected"]

print("=== Classification Report ===")
print(classification_report(y_true, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

# ==== 5. Визуализация ====
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, client in enumerate(client_ids[:3]):
    temp = df[df['client_id'] == client]
    churn_flag = temp['churn'].dropna().values[0]
    axes[i].plot(temp['month'], temp['amount'], marker='o')
    axes[i].set_title(f"Client {client} — churn: {churn_flag}")
    axes[i].axvspan(10, 12, color='red', alpha=0.1)  # последние 3 месяца

plt.tight_layout()
plt.show()
