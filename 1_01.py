import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sheet_names = ['202301', '202302', '202303', '202304', '202305', '202306',
               '202307', '202308', '202309', '202310', '202311', '202312']

xls = 'clients.xlsx'

total_clients = []
lost_clients = []
new_clients = []

all_seen_clients = set()

for i, sheet in enumerate(tqdm(sheet_names, desc="Обработка месяцев")):
    df = pd.read_excel(xls, sheet_name=sheet)
    clients = set(df['CLIENTBASENUMBER'].unique())
    total_clients.append(len(clients))
    lost_clients.append(df.loc[df['pr'] == True, 'CLIENTBASENUMBER'].nunique())
    if i == 0:
        new_clients.append(np.nan)
    else:
        new = clients - all_seen_clients
        new_clients.append(len(new))
    all_seen_clients.update(clients)

fig = make_subplots(rows=1, cols=3, subplot_titles=[
    "Общее число клиентов",
    "Ушедшие клиенты",
    "Пришедшие клиенты"
])

fig.add_trace(go.Bar(
    x=sheet_names,
    y=total_clients,
    name="Общее число клиентов",
    marker_color='steelblue',
    text=total_clients,
    textposition='outside'
), row=1, col=1)

fig.add_trace(go.Bar(
    x=sheet_names,
    y=lost_clients,
    name="Ушедшие клиенты",
    marker_color='indianred',
    text=lost_clients,
    textposition='outside'
), row=1, col=2)

fig.add_trace(go.Bar(
    x=sheet_names,
    y=new_clients,
    name="Пришедшие клиенты",
    marker_color='seagreen',
    text=[None if np.isnan(val) else val for val in new_clients],
    textposition='outside'
), row=1, col=3)

fig.update_layout(
    title={
        'text': "📊 Анализ клиентской базы по месяцам – 2023",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24, color='darkblue', family='Arial Black')
    },
    height=550,
    width=1300,
    showlegend=False,
    margin=dict(t=100)
)

fig.update_xaxes(tickangle=45)

fig.update_yaxes(range=[30000, 35000], row=1, col=1)
fig.update_yaxes(range=[30000, 35000], row=1, col=2)
fig.update_yaxes(range=[30000, 35000], row=1, col=3)

fig.show()
