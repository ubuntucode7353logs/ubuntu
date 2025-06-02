import seaborn as sns
import matplotlib.pyplot as plt

for col in percent_features:
    plt.figure(figsize=(6, 3))
    sns.violinplot(x='pr', y=col, data=df)
    plt.title(f"{col} vs pr")
    plt.tight_layout()
    plt.show()


percent_features = [
    'dcts1', 'dcts2', 'dcts3',
    'dctq1', 'dctq2', 'dctq3',
    'ddts1', 'ddts2', 'ddts3',
    'ddtq1', 'ddtq2', 'ddtq3',
    'dkc1', 'dkc2', 'dkc3',
    'DVOO1', 'DVOO2', 'DVOO3',
    'd13812', 'd23812', 'd33812',
    'drko1', 'drko2', 'drko3'
]
