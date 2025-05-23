import pandas as pd

def process_sheet(df):
    df['feat_otherbct'] = (df['otherbct'] > 0).astype(int)
    df['feat_newpmnt'] = (df['newpmnt'] > 0).astype(int)
    df['feat_kcneg'] = ((df['kcneg1'] > 0) | (df['kcneg2'] > 0) | (df['kcneg3'] > 0)).astype(int)
    df['feat_dkc'] = ((abs(df['dkc1']) >= 0.7) | (abs(df['dkc2']) >= 0.7) | (abs(df['dkc3']) >= 0.7)).astype(int)
    df['feat_dcts'] = ((abs(df['dcts1']) >= 0.5) | (abs(df['dcts2']) >= 0.5) | (abs(df['dcts3']) >= 0.5)).astype(int)
    df['feat_ddtq'] = ((abs(df['ddtq1']) >= 0.7) | (abs(df['ddtq2']) >= 0.7) | (abs(df['ddtq3']) >= 0.7)).astype(int)
    df['feat_deal_loss'] = df['deal1_reason'].str.contains("экономика", case=False, na=False).astype(int)
    df['feat_kciv'] = ((df['kciv1'] > 0) | (df['kciv2'] > 0) | (df['kciv3'] > 0)).astype(int)
    df['feat_inactive'] = ((df['act1'] == 1) | (df['act2'] == 1) | (df['act3'] == 1)).astype(int)
    df['feat_pmnt_issue'] = ((df['pmnt1'] > 0) | (df['pmnt2'] > 0) | (df['pmnt3'] > 0)).astype(int)
    df['feat_dvoo'] = ((abs(df['dvoo1']) >= 0.5) | (abs(df['dvoo2']) >= 0.5) | (abs(df['dvoo3']) >= 0.5)).astype(int)
    df['feat_zp_downtrend'] = ((df['zp1'] > df['zp2']) & (df['zp2'] > df['zp3'])).astype(int)
    df['feat_dividends'] = ((df['divid1'] > 0) | (df['divid2'] > 0) | (df['divid3'] > 0)).astype(int)
    df['feat_pack_change'] = ((df['PACK1'] == 1) | (df['PACK2'] == 1) | (df['PACK3'] == 1)).astype(int)

    df['churn_score'] = (
        df['feat_otherbct'] * 1.0 +
        df['feat_newpmnt'] * 1.0 +
        df['feat_kcneg'] * 0.3 +
        df['feat_dkc'] * 0.1 +
        df['feat_dcts'] * 0.7 +
        df['feat_ddtq'] * 0.1 +
        df['feat_deal_loss'] * 0.9 +
        df['feat_kciv'] * 0.2 +
        df['feat_inactive'] * 1.0 +
        df['feat_pmnt_issue'] * 0.2 +
        df['feat_dvoo'] * 0.8 +
        df['feat_zp_downtrend'] * 0.3 +
        df['feat_dividends'] * 0.1 +
        df['feat_pack_change'] * 0.3
    )

    df['high_risk'] = (df['churn_score'] >= 2.0).astype(int)
    return df[['CLIENTBASENUMBER', 'high_risk', 'churned']]

def evaluate_excel(file_path):
    all_results = []
    xls = pd.ExcelFile(file_path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if 'churned' not in df.columns:
            continue
        processed = process_sheet(df)
        processed['sheet'] = sheet
        all_results.append(processed)

    full_df = pd.concat(all_results, ignore_index=True)

    from sklearn.metrics import classification_report, confusion_matrix

    print(confusion_matrix(full_df['churned'], full_df['high_risk']))
    print(classification_report(full_df['churned'], full_df['high_risk'], digits=3))

    return full_df

# Пример вызова
# result_df = evaluate_excel("путь_к_файлу.xlsx")
