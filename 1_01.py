import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_per_sheet = {
    '202310': [{'feature': 'dcts1', 'percent_nonzero': 86.8},
               {'feature': 'dcts2', 'percent_nonzero': 90.4},
               {'feature': 'dcts3', 'percent_nonzero': 92.3},
               {'feature': 'dctq1', 'percent_nonzero': 78.1}],
    '202311': [{'feature': 'dcts1', 'percent_nonzero': 83.5},
               {'feature': 'dcts2', 'percent_nonzero': 86.7},
               {'feature': 'dcts3', 'percent_nonzero': 90.1},
               {'feature': 'dctq1', 'percent_nonzero': 74.0}],
    '202312': [{'feature': 'dcts1', 'percent_nonzero': 82.0},
               {'feature': 'dcts2', 'percent_nonzero': 85.0},
               {'feature': 'dcts3', 'percent_nonzero': 89.0},
               {'feature': 'dctq1', 'percent_nonzero': 72.0}]
}

def plot_percent_only(data_per_sheet):
    sheets = list(data_per_sheet.keys())
    n_sheets = len(sheets)
    ncols = 4
    nrows = (n_sheets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.flatten()

    for i, sheet in enumerate(sheets):
        data = data_per_sheet[sheet]
        df = pd.DataFrame(data)
        x = np.arange(len(df))
        width = 0.7

        ax = axes[i]
        bars = ax.bar(x, df['percent_nonzero'], width, color='orange')
        ax.set_title(f'{sheet} — Percent ≠ 0')
        ax.set_xticks(x)
        ax.set_xticklabels(df['feature'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Процент, %')
        ax.set_ylim(0, df['percent_nonzero'].max()*1.1)
        ax.grid(axis='y')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x()+bar.get_width()/2, height),
                        xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    # Удаляем лишние подграфики, если они есть
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_percent_only(data_per_sheet)
