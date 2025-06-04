from sklearn.inspection import permutation_importance

perm_importances_list = []

for i, sheet_name in enumerate(tqdm(sheet_names, desc="Обработка месяцев")):
    df_sheet = base.parse(sheet_name=sheet_name)
    df_sheet['pr'] = df_sheet['pr'].fillna(0)

    X_percent, X_bool = df_sheet[percent_features], df_sheet[bool_features]
    y = df_sheet['pr']

    X_percent_scaled = StandardScaler().fit_transform(X_percent)
    X_all = np.hstack([X_percent_scaled, X_bool.values])
    feature_names = percent_features + bool_features

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    roc_aucs.append(roc_auc)

    # Коэффициенты логрегрессии
    coefs = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    })
    coefs_list.append(coefs)

    # Permutation Importance
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc'
    )
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)
    perm_importances_list.append(perm_df)

    # Визуализация Permutation Importance
    top_perm = perm_df.head(15)
    sns.barplot(
        data=top_perm, x='importance_mean', y='feature', palette='viridis', ax=axes[i]
    )
    axes[i].set_title(f'Лист {sheet_name}\nPerm ROC AUC: {roc_auc:.3f}')
    axes[i].grid(True)


import shap

# 1. Загрузка нужного листа
df_sheet = base.parse('202312')
df_sheet['pr'] = df_sheet['pr'].fillna(0)

X_percent, X_bool = df_sheet[percent_features], df_sheet[bool_features]
y = df_sheet['pr']

# 2. Масштабирование и объединение
X_percent_scaled = StandardScaler().fit_transform(X_percent)
X_all = np.hstack([X_percent_scaled, X_bool.values])
feature_names = percent_features + bool_features
X_df = pd.DataFrame(X_all, columns=feature_names)

# 3. Трейн/тест
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, stratify=y, random_state=42
)

# 4. Модель
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# 5. SHAP объяснения
explainer = shap.Explainer(model, X_train, feature_names=feature_names)
shap_values = explainer(X_test)

# 6. Глобальное объяснение (summary plot)
shap.summary_plot(shap_values, X_test, max_display=15)
