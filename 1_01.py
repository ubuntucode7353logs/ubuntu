import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

data_per_sheet = {
    '202310': [{'feature': 'dcts1', 'nonzero_count': 27399, 'percent_nonzero': 86.8},
               {'feature': 'dcts2', 'nonzero_count': 28520, 'percent_nonzero': 90.4},
               {'feature': 'dcts3', 'nonzero_count': 29133, 'percent_nonzero': 92.3},
               {'feature': 'dctq1', 'nonzero_count': 24656, 'percent_nonzero': 78.1}],
    '202311': [{'feature': 'dcts1', 'nonzero_count': 26000, 'percent_nonzero': 83.5},
               {'feature': 'dcts2', 'nonzero_count': 27000, 'percent_nonzero': 86.7},
               {'feature': 'dcts3', 'nonzero_count': 28000, 'percent_nonzero': 90.1},
               {'feature': 'dctq1', 'nonzero_count': 23000, 'percent_nonzero': 74.0}],
    '202312': [{'feature': 'dcts1', 'nonzero_count': 25500, 'percent_nonzero': 82.0},
               {'feature': 'dcts2', 'nonzero_count': 26500, 'percent_nonzero': 85.0},
               {'feature': 'dcts3', 'nonzero_count': 27500, 'percent_nonzero': 89.0},
               {'feature': 'dctq1', 'nonzero_count': 22000, 'percent_nonzero': 72.0}]
}

def plot_nested_subplots(data_per_sheet):
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

        ax_main = axes[i]
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_main.get_subplotspec(), wspace=0.3)

        ax_count = fig.add_subplot(gs[0])
        bars_count = ax_count.bar(x, df['nonzero_count'], width, color='skyblue')
        ax_count.set_title(f'{sheet} — Count ≠ 0')
        ax_count.set_xticks(x)
        ax_count.set_xticklabels(df['feature'], rotation=45, ha='right', fontsize=8)
        ax_count.set_ylabel('Количество')
        ax_count.grid(axis='y')

        # Установка лимита Y с запасом сверху
        ymax_count = df['nonzero_count'].max()
        ax_count.set_ylim(0, ymax_count * 1.1)

        for bar in bars_count:
            height = bar.get_height()
            ax_count.annotate(f'{int(height)}', xy=(bar.get_x()+bar.get_width()/2, height),
                              xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

        ax_percent = fig.add_subplot(gs[1])
        bars_percent = ax_percent.bar(x, df['percent_nonzero'], width, color='orange')
        ax_percent.set_title(f'{sheet} — Percent ≠ 0')
        ax_percent.set_xticks(x)
        ax_percent.set_xticklabels(df['feature'], rotation=45, ha='right', fontsize=8)
        ax_percent.set_ylabel('Процент, %')
        ax_percent.grid(axis='y')

        # Установка лимита Y с запасом сверху
        ymax_percent = df['percent_nonzero'].max()
        ax_percent.set_ylim(0, ymax_percent * 1.1)

        for bar in bars_percent:
            height = bar.get_height()
            ax_percent.annotate(f'{height:.1f}%', xy=(bar.get_x()+bar.get_width()/2, height),
                               xytext=(0,3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

        fig.delaxes(ax_main)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_nested_subplots(data_per_sheet)
