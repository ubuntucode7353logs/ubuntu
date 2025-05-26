# Пример списков переменных
bool_vars = ['PACK1', 'PACK2', 'PACK3', 'act1', 'act2', 'act3', 'zp1', 'zp2', 'zp3',
             'newb', 'otherbdt', 'otherbct', 'newpmnt']

# Функция расчета процента ненулевых значений
def calc_nonzero_percentage_df(df, cols):
    data = []
    for col in cols:
        nonzero_count = (df[col] != 0).sum()
        total = df[col].notna().sum()
        percent = 100 * nonzero_count / total if total > 0 else np.nan
        data.append({'variable': col, 'nonzero_count': nonzero_count, 'total': total, 'nonzero_percent': percent})
    return pd.DataFrame(data).sort_values(by='nonzero_percent', ascending=False).reset_index(drop=True)

# Применение к булевым переменным
bool_nonzero_df = calc_nonzero_percentage_df(df_all, bool_vars)

# Вывод результата
bool_nonzero_df
